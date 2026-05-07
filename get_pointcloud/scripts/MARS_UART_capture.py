"""MARS UART 擷取工具：依照 MARS 論文格式錄製點雲並儲存 NPY。

輸出格式：
    (N, 8, 8, 5)

每個 frame 的處理流程：
    1. 從 UART TLV 取得點雲 [x, y, z, doppler, intensity]
    2. 不做 ROI 篩選，保留 out-of-range / ghost points
    3. 依 x -> y -> z 升冪排序
    4. 截斷或補零到 64 points
    5. row-major reshape 成 (8, 8, 5)

第 5 維 intensity 來源：
    TI xWRL6844 OOB point cloud 沒有直接輸出 MARS 論文中的 reflection intensity I_i，
    這裡使用 side-info SNR 作為 intensity channel 的可用近似值。
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np
import serial

import util.get_abs_dir as get_abs_dir


TLV_DETECTED_POINTS = 1
TLV_SIDE_INFO = 7
MAGIC_WORD = bytes([2, 1, 4, 3, 6, 5, 8, 7])

FRAME_LENGTH_OFFSET = 12
FRAME_HEADER_SIZE = 40
POINT_DATA_SIZE = 16
SIDE_INFO_DATA_SIZE = 4

DEFAULT_CONFIG_PORT = 'COM19'
DEFAULT_DATA_PORT = 'COM20'
DEFAULT_CFG_FILE = 'IWRL6844_4T4R_record.cfg'
DEFAULT_FRAMES = 200
DEFAULT_BAUDRATE_CFG = 115200
DEFAULT_BAUDRATE_DATA = 1250000

path_project_root, _, _ = get_abs_dir.get_abs_dir()
DEFAULT_CFG_PATH = os.path.join(path_project_root, 'get_pointcloud', 'cfg', DEFAULT_CFG_FILE)

file_class = 'test'  # 'standard' 'reference' 'test'
DEFAULT_OUTPUT_DIR = os.path.join(path_project_root, 'feature', file_class)

IntensityMode = str


@dataclass(frozen=True)
class CaptureSettings:
	cfg_path: str
	port_cfg: str
	port_data: str
	frames: int
	out_path: str
	baudrate_cfg: int = DEFAULT_BAUDRATE_CFG
	baudrate_data: int = DEFAULT_BAUDRATE_DATA
	intensity_mode: IntensityMode = 'snr_db'
	snr_norm_mean: float = 20.0
	snr_norm_std: float = 10.0
	filter_roi: bool = False
	dtype: str = 'float64'


@dataclass
class FrameData:
	frame_number: int
	total_length: int
	num_detected_objects: int
	num_tlv: int
	points: np.ndarray
	_debug_tlv_bytes: bytes | None = None


def resolve_cfg_path(cfg_path: str) -> str:
	"""支援只輸入檔名時，優先到專案 cfg 目錄找。"""
	if os.path.isabs(cfg_path) and os.path.isfile(cfg_path):
		return cfg_path

	candidates = [
		cfg_path,
		os.path.join(os.path.dirname(__file__), cfg_path),
		os.path.join(path_project_root, 'get_pointcloud', 'cfg', os.path.basename(cfg_path)),
	]

	for candidate in candidates:
		candidate = os.path.normpath(candidate)
		if os.path.isfile(candidate):
			return candidate

	return os.path.normpath(os.path.join(path_project_root, 'get_pointcloud', 'cfg', os.path.basename(cfg_path)))


def resolve_output_path(out_path: str) -> str:
	"""產生輸出路徑：預設 radar_capture_0.npy，若已存在則遞增。"""
	if not out_path:
		base_dir = DEFAULT_OUTPUT_DIR
		os.makedirs(base_dir, exist_ok=True)
		out_path = os.path.join(base_dir, 'radar_capture_0.npy')

	out_path = os.path.normpath(out_path)
	base, ext = os.path.splitext(out_path)
	if ext == '':
		ext = '.npy'
		out_path = base + ext

	if not os.path.isabs(out_path):
		out_path = os.path.normpath(os.path.join(os.getcwd(), out_path))

	if not os.path.exists(out_path):
		parent = os.path.dirname(out_path)
		if parent and not os.path.isdir(parent):
			os.makedirs(parent, exist_ok=True)
		return out_path

	name_root = base
	parts = base.rsplit('_', 1)
	if len(parts) == 2 and parts[1].isdigit():
		name_root = parts[0]

	i = 1
	while True:
		new_path = f'{name_root}_{i}{ext}'
		if not os.path.exists(new_path):
			parent = os.path.dirname(new_path)
			if parent and not os.path.isdir(parent):
				os.makedirs(parent, exist_ok=True)
			return new_path
		i += 1


def find_magic_word(buffer: bytearray, magic_word: bytes) -> int:
	return buffer.find(magic_word)


def side_info_to_db(raw_value: int) -> float:
	"""將 TLV side-info raw int16 轉成 dB。

	TI OOB side-info 通常是 1 LSB = 0.1 dB；若換算後明顯過大，
	使用 0.01 dB/LSB 修正，避免 167 dB 這類不合理顯示。
	"""
	value_db = float(raw_value) * 0.1
	if abs(value_db) > 100.0:
		value_db = float(raw_value) * 0.01
	return value_db


def convert_intensity(snr_raw: int, settings: CaptureSettings) -> float:
	"""把 side-info SNR 轉成第 5 維 intensity channel。"""
	if settings.intensity_mode == 'snr_raw':
		return float(snr_raw)

	snr_db = side_info_to_db(snr_raw)
	if settings.intensity_mode == 'snr_norm':
		if settings.snr_norm_std == 0:
			raise ValueError('snr_norm_std cannot be 0')
		return (snr_db - settings.snr_norm_mean) / settings.snr_norm_std

	return snr_db


def parse_frame(frame_bytes: bytes, settings: CaptureSettings) -> FrameData | None:
	"""解析完整 frame，回傳點雲與 frame header 資訊。"""
	if len(frame_bytes) < FRAME_HEADER_SIZE:
		return None

	_first_tlv_bytes = None
	offset = 8  # magic word
	try:
		offset += 4  # version
		total_length = struct.unpack_from('<I', frame_bytes, offset)[0]
		offset += 4  # totalLen
		offset += 4  # platform
		frame_number = struct.unpack_from('<I', frame_bytes, offset)[0]
		offset += 4  # frameNumber
		offset += 4  # timeCPUCycles
		num_detected_objects = struct.unpack_from('<I', frame_bytes, offset)[0]
		offset += 4
		num_tlv = struct.unpack_from('<I', frame_bytes, offset)[0]
		offset += 4
		offset += 4  # subFrameNumber
	except struct.error:
		return None

	if num_detected_objects == 0:
		return FrameData(
			frame_number=frame_number,
			total_length=total_length,
			num_detected_objects=0,
			num_tlv=num_tlv,
			points=np.zeros((0, 5), dtype=np.float64),
		)

	points = np.zeros((num_detected_objects, 5), dtype=np.float64)

	for _ in range(num_tlv):
		if offset + 8 > len(frame_bytes):
			break

		tlv_type, tlv_length = struct.unpack_from('<II', frame_bytes, offset)
		offset += 8
		tlv_end = offset + tlv_length

		if tlv_type == TLV_DETECTED_POINTS:
			for point_index in range(num_detected_objects):
				if offset + POINT_DATA_SIZE > len(frame_bytes):
					break
				if _first_tlv_bytes is None and point_index == 0:
					_first_tlv_bytes = bytes(frame_bytes[offset:offset + POINT_DATA_SIZE])
				points[point_index, 0:4] = struct.unpack_from('<ffff', frame_bytes, offset)
				offset += POINT_DATA_SIZE

		elif tlv_type == TLV_SIDE_INFO:
			for point_index in range(num_detected_objects):
				if offset + SIDE_INFO_DATA_SIZE > len(frame_bytes):
					break
				snr_raw, _noise_raw = struct.unpack_from('<hh', frame_bytes, offset)
				offset += SIDE_INFO_DATA_SIZE
				points[point_index, 4] = convert_intensity(snr_raw, settings)

		else:
			offset = tlv_end

		if offset < tlv_end:
			offset = tlv_end

	return FrameData(
		frame_number=frame_number,
		total_length=total_length,
		num_detected_objects=num_detected_objects,
		num_tlv=num_tlv,
		points=points,
		_debug_tlv_bytes=_first_tlv_bytes,
	)


def frame_to_featuremap(frame: np.ndarray, max_points: int = 64, filter_roi: bool = False, dtype=np.float64) -> np.ndarray:
	"""依 MARS 論文產生 (8,8,5) feature map。

	MARS 論文流程：
	- 每點 5 維：[x, y, z, Doppler, intensity]
	- 每 frame 統一為 64 x 5；不足補零，超過截斷
	- 依 x -> y -> z 升冪排序
	- row-major reshape 成 8 x 8 x 5

	預設不做 ROI filtering；這樣 out-of-range / ghost points 也會保留。
	"""
	if frame is None or frame.shape[0] == 0:
		return np.zeros((8, 8, 5), dtype=dtype)

	frame = frame.astype(dtype, copy=False)

	# 移除完整零列；有效點中若某一欄為 0 不會被移除。
	frame = frame[np.any(frame != 0, axis=1)]

	# 舊版相容選項：預設關閉。MARS 論文實驗中 out-of-range points 會保留。
	if filter_roi and frame.shape[0] > 0:
		mask = (
			(frame[:, 0] > -1.0) & (frame[:, 0] < 1.0) &
			(frame[:, 1] > 0.0) & (frame[:, 1] < 3.0) &
			(frame[:, 2] > -1.0) & (frame[:, 2] < 1.0)
		)
		frame = frame[mask]

	# 依 x -> y -> z 升冪排序。np.lexsort 最後一個 key 是 primary key。
	if frame.shape[0] > 0:
		frame = frame[np.lexsort((frame[:, 2], frame[:, 1], frame[:, 0]))]

	# 截斷到 64 points。
	if frame.shape[0] > max_points:
		frame = frame[:max_points]

	# 不足補零。
	if frame.shape[0] < max_points:
		pad = np.zeros((max_points - frame.shape[0], 5), dtype=dtype)
		frame = np.vstack((frame, pad))

	return frame.reshape(8, 8, 5)


class RadarUARTCapture:
	def __init__(self, settings: CaptureSettings):
		self.settings = settings
		self.cfg_port = serial.Serial(
			settings.port_cfg,
			baudrate=settings.baudrate_cfg,
			timeout=0.1,
		)
		self.data_port = serial.Serial(
			settings.port_data,
			baudrate=settings.baudrate_data,
			timeout=0.1,
		)
		self.byte_buffer = bytearray()

	def _send_cli(self, cmd: str, delay: float = 0.12) -> str:
		if not cmd.endswith('\n'):
			cmd += '\n'
		self.cfg_port.write(cmd.encode('utf-8'))
		self.cfg_port.flush()
		time.sleep(delay)
		resp = b''
		while self.cfg_port.in_waiting:
			resp += self.cfg_port.read(self.cfg_port.in_waiting)
			time.sleep(0.02)
		return resp.decode(errors='ignore').strip()

	def send_config(self) -> None:
		print('[INFO] 送出 config...')
		self.cfg_port.reset_input_buffer()
		self.cfg_port.reset_output_buffer()
		self.data_port.reset_input_buffer()
		self.data_port.reset_output_buffer()
		self.byte_buffer.clear()

		resp = self._send_cli('sensorStop 0', delay=0.8)
		if resp:
			print('[CLI]', resp)

		with open(self.settings.cfg_path, 'r', encoding='utf-8') as file_handle:
			for line in file_handle:
				line = line.strip()
				if not line or line.startswith('%') or line.startswith('#'):
					continue

				# 前面已經送過 sensorStop，避免重複。
				if line.startswith('sensorStop'):
					continue

				delay = 1.0 if line.startswith('sensorStart') else 0.12
				resp = self._send_cli(line, delay=delay)
				print(f'  [CFG] {line}')
				if resp:
					print('[CLI]', resp)
				if 'error' in resp.lower():
					print(f'[WARN] 這行 cfg 可能失敗: {line}')

		time.sleep(0.5)
		self.data_port.reset_input_buffer()
		self.byte_buffer.clear()
		print('[INFO] Config 送出完成。\n')

	def read_frame(self) -> FrameData | None:
		while True:
			n_avail = self.data_port.in_waiting
			if n_avail > 0:
				self.byte_buffer.extend(self.data_port.read(n_avail))

			if len(self.byte_buffer) > 65536:
				del self.byte_buffer[:-32768]

			magic_idx = find_magic_word(self.byte_buffer, MAGIC_WORD)
			if magic_idx < 0:
				time.sleep(0.005)
				continue

			if len(self.byte_buffer) < magic_idx + FRAME_LENGTH_OFFSET + 4:
				time.sleep(0.005)
				continue

			total_len = struct.unpack_from('<I', self.byte_buffer, magic_idx + FRAME_LENGTH_OFFSET)[0]
			if total_len < FRAME_HEADER_SIZE or total_len > 65535:
				del self.byte_buffer[:magic_idx + len(MAGIC_WORD)]
				continue

			if len(self.byte_buffer) < magic_idx + total_len:
				time.sleep(0.005)
				continue

			frame_bytes = bytes(self.byte_buffer[magic_idx:magic_idx + total_len])
			del self.byte_buffer[:magic_idx + total_len]

			frame_data = parse_frame(frame_bytes, self.settings)
			if frame_data is not None:
				return frame_data

	def stop_sensor(self) -> None:
		try:
			if self.cfg_port and self.cfg_port.is_open:
				resp = self._send_cli('sensorStop 0', delay=0.8)
				if resp:
					print('[INFO] sensorStop 回應:', resp)
		except Exception as exc:
			print(f'[WARN] sensorStop 送出失敗: {exc}')

	def close(self) -> None:
		try:
			self.stop_sensor()
		finally:
			try:
				if self.data_port and self.data_port.is_open:
					self.data_port.reset_input_buffer()
					self.data_port.close()
			finally:
				if self.cfg_port and self.cfg_port.is_open:
					self.cfg_port.close()


def capture_to_npy(settings: CaptureSettings) -> int:
	print(f'[INFO] Config 檔案: {settings.cfg_path}')
	print(f'[INFO] Config serial port: {settings.port_cfg}')
	print(f'[INFO] Data serial port: {settings.port_data}')
	print(f'[INFO] 輸出檔案: {settings.out_path}')
	print(f'[INFO] intensity_mode: {settings.intensity_mode}')
	print(f'[INFO] filter_roi: {settings.filter_roi}')
	print(f'[INFO] dtype: {settings.dtype}')

	dtype = np.float64 if settings.dtype == 'float64' else np.float32
	fmaps: list[np.ndarray] = []
	capture = None

	try:
		capture = RadarUARTCapture(settings)
		capture.send_config()

		print(f'[INFO] 開始錄製，目標 {settings.frames} frames。')

		frame_count = 0
		while settings.frames < 0 or frame_count < settings.frames:
			frame_data = capture.read_frame()
			if frame_data is None:
				continue

			frame_count += 1
			valid_points = frame_data.points[np.any(frame_data.points != 0, axis=1)]

			fmap = frame_to_featuremap(
				valid_points,
				max_points=64,
				filter_roi=settings.filter_roi,
				dtype=dtype,
			)
			fmaps.append(fmap.astype(dtype, copy=False))

			if frame_count % 10 == 0:
				nonzero_points = np.count_nonzero(np.any(fmap.reshape(-1, 5) != 0, axis=1))
				print(f'[INFO] 已擷取 {frame_count} frames, 本幀有效點 {nonzero_points}, 累積 featuremap {len(fmaps)}')

	except KeyboardInterrupt:
		print('\n[INFO] 使用者中斷。')
	finally:
		if capture is not None:
			capture.close()

	if fmaps:
		captured = np.stack(fmaps).astype(dtype, copy=False)
	else:
		captured = np.zeros((0, 8, 8, 5), dtype=dtype)

	np.save(settings.out_path, captured)
	print(f'[INFO] 已儲存 featuremap NPY: {settings.out_path}')
	print(f'[INFO] 資料維度: {captured.shape} (frames,8,8,5)')
	print(f'[INFO] dtype: {captured.dtype}')

	if captured.size > 0:
		for ch in range(captured.shape[-1]):
			data = captured[..., ch]
			print(
				f'[INFO] channel {ch}: '
				f'min={np.min(data):.6f}, max={np.max(data):.6f}, '
				f'mean={np.mean(data):.6f}, std={np.std(data):.6f}, '
				f'nonzero={np.count_nonzero(data)}'
			)

	return 0


def main() -> int:
	parser = argparse.ArgumentParser(description='MARS UART 擷取並輸出 MARS 論文格式 NPY')
	parser.add_argument('--cfg', default=DEFAULT_CFG_PATH, help='Config 檔案路徑')
	parser.add_argument('--port_cfg', default=DEFAULT_CONFIG_PORT, help='Config serial port')
	parser.add_argument('--port_data', default=DEFAULT_DATA_PORT, help='Data serial port')
	parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES, help='要擷取的 frame 數，-1 代表持續錄製')
	parser.add_argument('--out', default='', help='輸出 NPY 檔案路徑')
	parser.add_argument('--baudrate_cfg', type=int, default=DEFAULT_BAUDRATE_CFG, help='Config serial baudrate')
	parser.add_argument('--baudrate_data', type=int, default=DEFAULT_BAUDRATE_DATA, help='Data serial baudrate')
	parser.add_argument(
		'--intensity_mode',
		choices=['snr_db', 'snr_raw', 'snr_norm'],
		default='snr_db',
		help='第 5 維 intensity 的來源：snr_db=接近論文 Intensity；snr_raw=原始 side-info；snr_norm=給舊模型近似標準化',
	)
	parser.add_argument('--snr_norm_mean', type=float, default=20.0, help='snr_norm 模式使用的 mean')
	parser.add_argument('--snr_norm_std', type=float, default=10.0, help='snr_norm 模式使用的 std')
	parser.add_argument('--filter_roi', action='store_true', help='啟用舊版 ROI 篩選；預設關閉以符合 MARS 論文保留 out-of-range points')
	parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64', help='輸出 NPY dtype；模型訓練檔常見為 float64')
	args = parser.parse_args()

	cfg_path = resolve_cfg_path(args.cfg)
	if not os.path.isfile(cfg_path):
		print(f'[ERROR] Config 檔案不存在: {cfg_path}')
		return 1

	out_path = resolve_output_path(args.out)

	settings = CaptureSettings(
		cfg_path=cfg_path,
		port_cfg=args.port_cfg,
		port_data=args.port_data,
		frames=args.frames,
		out_path=out_path,
		baudrate_cfg=args.baudrate_cfg,
		baudrate_data=args.baudrate_data,
		intensity_mode=args.intensity_mode,
		snr_norm_mean=args.snr_norm_mean,
		snr_norm_std=args.snr_norm_std,
		filter_roi=args.filter_roi,
		dtype=args.dtype,
	)
	return capture_to_npy(settings)


if __name__ == '__main__':
	sys.exit(main())
