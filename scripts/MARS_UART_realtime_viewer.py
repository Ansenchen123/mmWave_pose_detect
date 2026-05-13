"""MARS UART 命令列顯示工具。

流程：
1. 開啟 cfg / data serial ports
2. 送出 cfg 檔
3. 讀取 UART frame
4. 解析 TLV 點雲與 side info
5. 直接把每個 frame 的內容印到 cmd
"""


#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from radar_uart import filter_points
from radar_uart import FrameData
from radar_uart import PointFilterSettings
from radar_uart import point_filter_settings_from_config
from radar_uart import RadarUARTCapture
from radar_uart import RadarUARTSettings
from radar_uart import select_mars_points
from util.AbsDir import AbsDir
from util.radar_config import cfg_get
from util.radar_config import cfg_range
from util.radar_config import load_radar_config
from util.radar_config import resolve_cfg_path

RADAR_CONFIG = load_radar_config()
POINT_FILTER_SETTINGS = point_filter_settings_from_config(RADAR_CONFIG)

DEFAULT_CONFIG_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'config_port', default='COM3'))
DEFAULT_DATA_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'data_port', default='COM4'))
DEFAULT_CFG_FILE = str(cfg_get(RADAR_CONFIG, 'radar', 'cfg_file', default='IWRL6844_4T4R_record.cfg'))
DEFAULT_FRAMES = int(cfg_get(RADAR_CONFIG, 'radar', 'frames', default=200))
DEFAULT_BAUDRATE_CFG = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_cfg', default=115200))
DEFAULT_BAUDRATE_DATA = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_data', default=1250000))
INTENSITY_MODE = str(cfg_get(RADAR_CONFIG, 'point_output', 'intensity_mode', default='snr_db'))
SNR_NORM_MEAN = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_mean', default=20.0))
SNR_NORM_STD = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_std', default=10.0))
SIDE_INFO_DB_LIMIT = float(cfg_get(RADAR_CONFIG, 'point_output', 'side_info_db_limit', default=100.0))
MAX_POINTS = int(cfg_get(RADAR_CONFIG, 'feature_map', 'max_points', default=64))
TRUNCATE_BEFORE_SORT = bool(cfg_get(RADAR_CONFIG, 'feature_map', 'truncate_before_sort', default=True))
DISPLAY_X, DISPLAY_Y, DISPLAY_Z = cfg_range(RADAR_CONFIG, 'point_cloud')

abs_dir = AbsDir()
path_project_root = abs_dir.path_project_root
DEFAULT_CFG_PATH = os.path.join(path_project_root, 'cfg', DEFAULT_CFG_FILE)


@dataclass
class CaptureSettings:
	cfg_path: str
	port_cfg: str
	port_data: str
	frames: int
	baudrate_cfg: int
	baudrate_data: int
	intensity_mode: str = 'snr_db'
	snr_norm_mean: float = 20.0
	snr_norm_std: float = 10.0
	side_info_db_limit: float = 100.0
	point_filter: PointFilterSettings = POINT_FILTER_SETTINGS
	max_points: int = 64
	truncate_before_sort: bool = True
	display_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (DISPLAY_X, DISPLAY_Y, DISPLAY_Z)
	plot_hz: float = 10.0
	print_points: bool = False


def format_point_line(point_index: int, point: np.ndarray) -> str:
	x, y, z, doppler, intensity = point
	return f'{point_index:02d}: x={x: .4f}, y={y: .4f}, z={z: .4f}, doppler={doppler: .4f}, intensity={intensity: .2f}'


def print_frame(frame_data: FrameData, display_index: int, settings: CaptureSettings) -> None:
	valid_points = filter_points(frame_data.points, settings.point_filter)
	valid_points = select_mars_points(
		valid_points,
		max_points=settings.max_points,
		truncate_before_sort=settings.truncate_before_sort,
	)

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

def points_for_display(frame_data: FrameData, settings: CaptureSettings) -> np.ndarray:
	pts = filter_points(frame_data.points, settings.point_filter)
	pts = select_mars_points(
		pts,
		max_points=settings.max_points,
		truncate_before_sort=settings.truncate_before_sort,
	)
	if pts.size == 0:
		return pts

	(x_min, x_max), (y_min, y_max), (z_min, z_max) = settings.display_ranges
	mask = (
		(pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
		(pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
		(pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
	)
	return pts[mask]


def range_center(value_range: tuple[float, float]) -> float:
	return (value_range[0] + value_range[1]) * 0.5


def range_span(value_range: tuple[float, float]) -> float:
	return max(value_range[1] - value_range[0], 1e-6)


def axis_reference(value_range: tuple[float, float], preferred: float = 0.0) -> float:
	return float(np.clip(preferred, value_range[0], value_range[1]))


def tick_values(value_range: tuple[float, float]) -> np.ndarray:
	span = range_span(value_range)
	if span <= 2.0:
		step = 0.5
	elif span <= 5.0:
		step = 1.0
	else:
		step = 2.0

	start = np.ceil(value_range[0] / step) * step
	stop = np.floor(value_range[1] / step) * step
	values = np.arange(start, stop + step * 0.5, step)
	return values[(values >= value_range[0] - 1e-9) & (values <= value_range[1] + 1e-9)]


def add_gl_line(view, gl, points, color, width: float = 1.0, mode: str = 'lines'):
	item = gl.GLLinePlotItem(
		pos=np.asarray(points, dtype=np.float32),
		color=color,
		width=width,
		mode=mode,
		antialias=True,
	)
	view.addItem(item)
	return item


def add_gl_text(view, gl, qt_gui, pos, text: str, color, size: int = 10):
	font = qt_gui.QFont('Helvetica', size)
	item = gl.GLTextItem(
		pos=np.asarray(pos, dtype=np.float32),
		text=text,
		color=color,
		font=font,
	)
	view.addItem(item)
	return item


def add_axis_guides(view, gl, qt_gui, display_ranges) -> None:
	(x_min, x_max), (y_min, y_max), (z_min, z_max) = display_ranges
	x_ref = axis_reference((x_min, x_max))
	y_ref = axis_reference((y_min, y_max))
	z_ref = axis_reference((z_min, z_max))
	x_span = range_span((x_min, x_max))
	y_span = range_span((y_min, y_max))
	z_span = range_span((z_min, z_max))
	tick_len = max(min(x_span, y_span, z_span) * 0.035, 0.04)
	axis_width = 2.5
	tick_width = 1.6
	box_color = (0.55, 0.55, 0.55, 0.45)
	tick_color = (0.9, 0.9, 0.9, 0.9)
	x_color = (1.0, 0.25, 0.25, 1.0)
	y_color = (0.25, 1.0, 0.35, 1.0)
	z_color = (0.3, 0.55, 1.0, 1.0)

	# Bounding box makes depth and scale readable when the view is rotated.
	corners = [
		(x_min, y_min, z_min), (x_max, y_min, z_min),
		(x_min, y_max, z_min), (x_max, y_max, z_min),
		(x_min, y_min, z_max), (x_max, y_min, z_max),
		(x_min, y_max, z_max), (x_max, y_max, z_max),
	]
	edges = [
		(corners[0], corners[1]), (corners[2], corners[3]),
		(corners[4], corners[5]), (corners[6], corners[7]),
		(corners[0], corners[2]), (corners[1], corners[3]),
		(corners[4], corners[6]), (corners[5], corners[7]),
		(corners[0], corners[4]), (corners[1], corners[5]),
		(corners[2], corners[6]), (corners[3], corners[7]),
	]
	add_gl_line(view, gl, [point for edge in edges for point in edge], box_color, width=1.0)

	add_gl_line(view, gl, [(x_min, y_ref, z_ref), (x_max, y_ref, z_ref)], x_color, width=axis_width)
	add_gl_line(view, gl, [(x_ref, y_min, z_ref), (x_ref, y_max, z_ref)], y_color, width=axis_width)
	add_gl_line(view, gl, [(x_ref, y_ref, z_min), (x_ref, y_ref, z_max)], z_color, width=axis_width)

	tick_segments = []
	for x in tick_values((x_min, x_max)):
		tick_segments.extend([(x, y_ref - tick_len, z_ref), (x, y_ref + tick_len, z_ref)])
	for y in tick_values((y_min, y_max)):
		tick_segments.extend([(x_ref - tick_len, y, z_ref), (x_ref + tick_len, y, z_ref)])
	for z in tick_values((z_min, z_max)):
		tick_segments.extend([(x_ref - tick_len, y_ref, z), (x_ref + tick_len, y_ref, z)])
	add_gl_line(view, gl, tick_segments, tick_color, width=tick_width)

	for x in tick_values((x_min, x_max)):
		add_gl_text(view, gl, qt_gui, (x, y_ref - tick_len * 4.0, z_ref - tick_len * 2.0), f'{x:g}', tick_color)
	for y in tick_values((y_min, y_max)):
		add_gl_text(view, gl, qt_gui, (x_ref - tick_len * 4.5, y, z_ref - tick_len * 2.0), f'{y:g}', tick_color)
	for z in tick_values((z_min, z_max)):
		add_gl_text(view, gl, qt_gui, (x_ref - tick_len * 5.0, y_ref - tick_len * 2.0, z), f'{z:g}', tick_color)

	add_gl_text(view, gl, qt_gui, (x_max + x_span * 0.06, y_ref, z_ref), 'X horiz (m)', x_color, size=12)
	add_gl_text(view, gl, qt_gui, (x_ref, y_max + y_span * 0.06, z_ref), 'Y depth (m)', y_color, size=12)
	add_gl_text(view, gl, qt_gui, (x_ref, y_ref, z_max + z_span * 0.08), 'Z height (m)', z_color, size=12)


def init_plot(settings: CaptureSettings):
	try:
		import pyqtgraph as pg
		import pyqtgraph.opengl as gl
		from pyqtgraph.Qt import QtWidgets
		from pyqtgraph.Qt import QtGui
	except ImportError as exc:
		raise RuntimeError(
			'需要先安裝 pyqtgraph 3D 依賴：python -m pip install pyqtgraph PyQt5 PyOpenGL'
		) from exc

	app = QtWidgets.QApplication.instance() or pg.mkQApp('MARS UART Realtime Viewer')
	(x_min, x_max), (y_min, y_max), (z_min, z_max) = settings.display_ranges
	x_center = range_center((x_min, x_max))
	y_center = range_center((y_min, y_max))
	z_center = range_center((z_min, z_max))
	x_span = range_span((x_min, x_max))
	y_span = range_span((y_min, y_max))
	z_span = range_span((z_min, z_max))

	view = gl.GLViewWidget()
	view.setWindowTitle('Realtime points 3D (pyqtgraph OpenGL)')
	view.setBackgroundColor((18, 20, 24))
	view.setCameraPosition(distance=max(x_span, y_span, z_span) * 1.8, elevation=20, azimuth=-62)
	view.opts['center'].setX(x_center)
	view.opts['center'].setY(y_center)
	view.opts['center'].setZ(z_center)

	grid = gl.GLGridItem()
	grid.setSize(x=x_span, y=y_span)
	grid.setSpacing(x=max(x_span / 8.0, 0.25), y=max(y_span / 8.0, 0.25))
	grid.translate(x_center, y_center, z_min)
	view.addItem(grid)

	add_axis_guides(view, gl, QtGui, settings.display_ranges)

	scatter = gl.GLScatterPlotItem(
		pos=np.empty((0, 3), dtype=np.float32),
		color=np.empty((0, 4), dtype=np.float32),
		size=8,
		pxMode=True,
	)
	view.addItem(scatter)
	view.show()
	app.processEvents()
	return {'app': app, 'view': view, 'scatter': scatter}


def y_to_rgba(y_values: np.ndarray, y_range: tuple[float, float]) -> np.ndarray:
	if y_values.size == 0:
		return np.empty((0, 4), dtype=np.float32)
	y_min, y_max = y_range
	t = np.clip((y_values - y_min) / range_span(y_range), 0.0, 1.0).astype(np.float32)
	return np.column_stack((
		0.2 + 0.7 * t,
		0.9 - 0.5 * t,
		1.0 - 0.8 * t,
		np.ones_like(t),
	)).astype(np.float32)


def update_plot(frame_data: FrameData, plot, settings: CaptureSettings):
	pts = points_for_display(frame_data, settings)
	scatter = plot['scatter']
	if pts.size == 0:
		scatter.setData(
			pos=np.empty((0, 3), dtype=np.float32),
			color=np.empty((0, 4), dtype=np.float32),
		)
	else:
		scatter.setData(
			pos=pts[:, 0:3].astype(np.float32, copy=False),
			color=y_to_rgba(pts[:, 1], settings.display_ranges[1]),
			size=8,
			pxMode=True,
		)
	plot['app'].processEvents()


def process_plot_events(plot):
	plot['app'].processEvents()

def capture_and_print(settings: CaptureSettings) -> int:
	print(f'[INFO] Config 檔案: {settings.cfg_path}')
	print(f'[INFO] Config serial port: {settings.port_cfg}')
	print(f'[INFO] Data serial port: {settings.port_data}')
	print(f'[INFO] intensity_mode: {settings.intensity_mode}')
	print(f'[INFO] filter_roi: {settings.point_filter.roi_enabled}')
	print(f'[INFO] display_ranges: X{settings.display_ranges[0]} Y{settings.display_ranges[1]} Z{settings.display_ranges[2]}')
	print(f'[INFO] plot_backend: pyqtgraph OpenGL')
	print(f'[INFO] plot_hz: {settings.plot_hz}')
	print(f'[INFO] print_points: {settings.print_points}')

	capture = None
	try:
		uart_settings = RadarUARTSettings(
			cfg_path=settings.cfg_path,
			port_cfg=settings.port_cfg,
			port_data=settings.port_data,
			baudrate_cfg=settings.baudrate_cfg,
			baudrate_data=settings.baudrate_data,
			intensity_mode=settings.intensity_mode,
			snr_norm_mean=settings.snr_norm_mean,
			snr_norm_std=settings.snr_norm_std,
			side_info_db_limit=settings.side_info_db_limit,
		)
		capture = RadarUARTCapture(uart_settings)
		capture.send_config()

		plot = init_plot(settings)

		print(f'[INFO] 開始錄製，目標 {settings.frames} frames。')
		if settings.print_points:
			print('[INFO] 內容會直接顯示在 cmd 中。\n')

		frame_count = 0
		plot_interval_s = 1.0 / settings.plot_hz if settings.plot_hz > 0 else 0.0
		next_plot_at = 0.0
		while settings.frames < 0 or frame_count < settings.frames:
			frame_data = capture.read_frame(timeout_s=0.02)
			if frame_data is None:
				process_plot_events(plot)
				continue

			frame_count += 1
			if settings.print_points:
				print_frame(frame_data, frame_count, settings)

			now = time.monotonic()
			if now >= next_plot_at:
				update_plot(frame_data, plot, settings)
				next_plot_at = now + plot_interval_s

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
	parser.add_argument('--plot_hz', type=float, default=10.0, help='繪圖更新頻率；10 約等於每 100 ms 更新一次')
	parser.add_argument('--print_points', action='store_true', help='逐 frame 印出點雲內容；會降低即時顯示流暢度')
	parser.add_argument(
		'--intensity_mode',
		choices=['snr_db', 'snr_raw', 'snr_norm'],
		default=INTENSITY_MODE,
		help='第 5 維 intensity 的來源',
	)
	parser.add_argument('--filter_roi', action='store_true', default=POINT_FILTER_SETTINGS.roi_enabled, help='啟用 ROI 篩選；預設跟 cfg/radar_uart_config.yaml 一致')
	parser.add_argument('--no_filter_roi', action='store_false', dest='filter_roi', help='關閉 ROI 篩選')
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
		intensity_mode=args.intensity_mode,
		snr_norm_mean=SNR_NORM_MEAN,
		snr_norm_std=SNR_NORM_STD,
		side_info_db_limit=SIDE_INFO_DB_LIMIT,
		point_filter=replace(POINT_FILTER_SETTINGS, roi_enabled=args.filter_roi),
		max_points=MAX_POINTS,
		truncate_before_sort=TRUNCATE_BEFORE_SORT,
		display_ranges=(DISPLAY_X, DISPLAY_Y, DISPLAY_Z),
		plot_hz=args.plot_hz,
		print_points=args.print_points,
	)
	return capture_and_print(settings)


if __name__ == '__main__':
	sys.exit(main())
