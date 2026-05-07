"""MARS UART 命令列顯示工具。

流程：
1. 開啟 cfg / data serial ports
2. 送出 cfg 檔
3. 讀取 UART frame
4. 解析 TLV 點雲與 side info
5. 直接把每個 frame 的內容印到 cmd
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
import matplotlib.pyplot as plt
from matplotlib import cm

import util.get_abs_dir as get_abs_dir


TLV_DETECTED_POINTS = 1
TLV_SIDE_INFO = 7
MAGIC_WORD = bytes([2, 1, 4, 3, 6, 5, 8, 7])

FRAME_LENGTH_OFFSET = 12
FRAME_HEADER_SIZE = 40
POINT_DATA_SIZE = 16
SIDE_INFO_DATA_SIZE = 4

DEFAULT_CONFIG_PORT = 'COM3'
DEFAULT_DATA_PORT = 'COM4'
# DEFAULT_CFG_FILE = 'IWRL6844_4T4R_realtime.cfg'
DEFAULT_CFG_FILE = 'IWRL6844_4T4R_record.cfg'
DEFAULT_FRAMES = -1		# 預設 -1 代表無限擷取，直到使用者中斷 (Ctrl+C)
DEFAULT_BAUDRATE_CFG = 115200
DEFAULT_BAUDRATE_DATA = 1250000


path_project_root, _, _ = get_abs_dir.get_abs_dir()
DEFAULT_CFG_PATH = os.path.join(path_project_root, 'get_pointcloud', 'cfg', DEFAULT_CFG_FILE)


@dataclass(frozen=True)
class CaptureSettings:
	cfg_path: str
	port_cfg: str
	port_data: str
	frames: int
	baudrate_cfg: int = DEFAULT_BAUDRATE_CFG
	baudrate_data: int = DEFAULT_BAUDRATE_DATA


@dataclass
class FrameData:
	frame_number: int
	total_length: int
	num_detected_objects: int
	num_tlv: int
	points: np.ndarray
	_debug_tlv_bytes: bytes = None  # 第一個 TLV point 的原始 16 bytes，用於診斷


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


def find_magic_word(buffer: bytearray, magic_word: bytes) -> int:
	return buffer.find(magic_word)


def parse_frame(frame_bytes: bytes) -> FrameData | None:
	"""解析完整 frame，回傳點雲與 frame header 資訊。"""
	if len(frame_bytes) < FRAME_HEADER_SIZE:
		return None

	_first_tlv_bytes = None  # 用於診斷 TLV 格式
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
			points=np.zeros((0, 5), dtype=np.float32),
		)

	points = np.zeros((num_detected_objects, 5), dtype=np.float32)

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
				snr = struct.unpack_from('<H', frame_bytes, offset)[0]
				offset += SIDE_INFO_DATA_SIZE
				points[point_index, 4] = float(snr) * 0.1

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

	def send_config(self) -> None:
		print('[INFO] 送出 config...')
		with open(self.settings.cfg_path, 'r', encoding='utf-8') as file_handle:
			for line in file_handle:
				line = line.strip()
				if not line or line.startswith('%'):
					continue
				self.cfg_port.write((line + '\n').encode('utf-8'))
				time.sleep(0.05)
				print(f'  [CFG] {line}')
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
			if len(self.byte_buffer) < magic_idx + total_len:
				time.sleep(0.005)
				continue

			frame_bytes = bytes(self.byte_buffer[magic_idx:magic_idx + total_len])
			del self.byte_buffer[:magic_idx + total_len]

			frame_data = parse_frame(frame_bytes)
			if frame_data is not None:
				return frame_data

	def close(self) -> None:
		try:
			self.cfg_port.close()
		finally:
			self.data_port.close()


def format_point_line(point_index: int, point: np.ndarray) -> str:
	x, y, z, doppler, snr = point
	return f'{point_index:02d}: x={x: .4f}, y={y: .4f}, z={z: .4f}, doppler={doppler: .4f}, snr={snr: .2f}'


def print_frame(frame_data: FrameData, display_index: int) -> None:
	valid_mask = np.any(frame_data.points != 0, axis=1)
	valid_points = frame_data.points[valid_mask]

	print()
	print(f'========== Frame {display_index} ==========' )
	print(f'frame_number        : {frame_data.frame_number}')
	print(f'total_length        : {frame_data.total_length}')
	print(f'num_detected_objects: {frame_data.num_detected_objects}')
	print(f'num_tlv             : {frame_data.num_tlv}')
	print(f'有效點數            : {len(valid_points)}')

	if len(valid_points) == 0:
		print('(no points)')
		return

	for idx, point in enumerate(valid_points, start=1):
		print(format_point_line(idx, point))

	# 診斷：印出原始 TLV float 值
	if frame_data._debug_tlv_bytes is not None and len(frame_data._debug_tlv_bytes) == 16:
		raw_floats = struct.unpack_from('<ffff', frame_data._debug_tlv_bytes, 0)
		print(f'\n[原始 TLV] 第一個點的四個 float: x={raw_floats[0]:.6f}, y={raw_floats[1]:.6f}, z={raw_floats[2]:.6f}, doppler={raw_floats[3]:.6f}')

	# 診斷：如果 x/z 都是 0，嘗試其他欄位排列
	if len(valid_points) > 0 and frame_data._debug_tlv_bytes is not None:
		first_x = valid_points[0, 0]
		first_z = valid_points[0, 2]
		if abs(first_x) < 1e-6 and abs(first_z) < 1e-6:
			print('\n[診斷] 檢測到 x/z 恆為 0，嘗試其他 TLV 欄位排列：')
			raw_bytes = frame_data._debug_tlv_bytes
			if len(raw_bytes) == 16:
				# 試試看四種常見排列
				interpretations = [
					('目前解析 [x, y, z, doppler]', struct.unpack_from('<ffff', raw_bytes, 0)),
					('試試 [y, x, z, doppler]', (
						struct.unpack_from('<f', raw_bytes, 4)[0],
						struct.unpack_from('<f', raw_bytes, 0)[0],
						struct.unpack_from('<f', raw_bytes, 8)[0],
						struct.unpack_from('<f', raw_bytes, 12)[0],
					)),
					('試試 [range, angle, elevation, doppler]', struct.unpack_from('<ffff', raw_bytes, 0)),
					('試試 [angle, range, elevation, doppler]', (
						struct.unpack_from('<f', raw_bytes, 4)[0],
						struct.unpack_from('<f', raw_bytes, 0)[0],
						struct.unpack_from('<f', raw_bytes, 8)[0],
						struct.unpack_from('<f', raw_bytes, 12)[0],
					)),
				]
				for desc, vals in interpretations:
					print(f'  {desc}: f1={vals[0]: .6f}, f2={vals[1]: .6f}, f3={vals[2]: .6f}, f4={vals[3]: .6f}')

def init_plot():
	plt.ion()
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.set_title('Realtime points (X-Z平面，用Y距離著色)')
	ax.set_xlabel('X 水平位置 (m)  ±1')
	ax.set_ylabel('Z 高度 (m)  ±1')

	# 固定範圍：X = ±1 (水平), Z = ±1 (高度)
	ax.set_xlim(-1.0, 1.0)
	ax.set_ylim(-1.0, 1.0)
	ax.set_aspect('equal', 'box')

	sc = ax.scatter([], [], c=[], cmap=cm.viridis, s=20, vmin=0.0, vmax=3.0)
	cbar = fig.colorbar(sc, ax=ax, pad=0.02, label='Y 距離深度 (m)')
	return fig, ax, sc, cbar


def update_plot(frame_data, ax, sc, cbar, vmin=0.0, vmax=None):
	valid_mask = np.any(frame_data.points != 0, axis=1)
	pts = frame_data.points[valid_mask]

	ax.set_xlim(-1.0, 1.0)
	ax.set_ylim(-1.0, 1.0)
	ax.set_aspect('equal', 'box')

	if pts.size == 0:
		sc.set_offsets(np.empty((0, 2)))
		sc.set_array(np.array([]))
		ax.set_title(f'Frame {frame_data.frame_number} (no points)')
	else:
		x = pts[:, 0]  # 水平
		y_dist = pts[:, 1]  # 距離（用作顏色）
		z = pts[:, 2]  # 高度

		sc.set_offsets(np.c_[x, z])  # X-Z 平面
		sc.set_array(y_dist)  # 用 Y 距離著色

		if vmax is None:
			clim_min = 0.0
			clim_max = 3.0
		else:
			clim_min, clim_max = vmin, vmax
		sc.set_clim(clim_min, clim_max)
		cbar.update_normal(sc)

		ax.set_title(f'Frame {frame_data.frame_number} - pts:{len(x)}')

	plt.draw()
	plt.pause(0.001)

def capture_and_print(settings: CaptureSettings) -> int:
	print(f'[INFO] Config 檔案: {settings.cfg_path}')
	print(f'[INFO] Config serial port: {settings.port_cfg}')
	print(f'[INFO] Data serial port: {settings.port_data}')

	capture = None
	try:
		capture = RadarUARTCapture(settings)
		capture.send_config()

		fig, ax, sc, cbar = init_plot()

		print(f'[INFO] 開始錄製，目標 {settings.frames} frames。')
		print('[INFO] 內容會直接顯示在 cmd 中。\n')

		frame_count = 0
		while settings.frames < 0 or frame_count < settings.frames:
			frame_data = capture.read_frame()
			if frame_data is None:
				continue

			frame_count += 1
			print_frame(frame_data, frame_count)
			update_plot(frame_data, ax, sc, cbar)

	except KeyboardInterrupt:
		print('\n[INFO] 使用者中斷。')
	finally:
		if capture is not None:
			capture.close()

	print('\n[INFO] 結束。')
	return 0


def main() -> int:
	parser = argparse.ArgumentParser(description='MARS UART 內容直接輸出到 cmd')
	parser.add_argument('--cfg', default=DEFAULT_CFG_PATH, help='Config 檔案路徑')
	parser.add_argument('--port_cfg', default=DEFAULT_CONFIG_PORT, help='Config serial port')
	parser.add_argument('--port_data', default=DEFAULT_DATA_PORT, help='Data serial port')
	parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES, help='要擷取的 frame 數')
	parser.add_argument('--baudrate_cfg', type=int, default=DEFAULT_BAUDRATE_CFG, help='Config serial baudrate')
	parser.add_argument('--baudrate_data', type=int, default=DEFAULT_BAUDRATE_DATA, help='Data serial baudrate')
	args = parser.parse_args()

	cfg_path = resolve_cfg_path(args.cfg)
	if not os.path.isfile(cfg_path):
		print(f'[ERROR] Config 檔案不存在: {cfg_path}')
		return 1

	settings = CaptureSettings(
		cfg_path=cfg_path,
		port_cfg=args.port_cfg,
		port_data=args.port_data,
		frames=args.frames,
		baudrate_cfg=args.baudrate_cfg,
		baudrate_data=args.baudrate_data,
	)
	return capture_and_print(settings)


if __name__ == '__main__':
	sys.exit(main())