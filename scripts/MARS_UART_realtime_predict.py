# -*- coding: utf-8 -*-
"""Realtime UART point-cloud capture + MARS pose estimation.

流程：
1. 送出 radar cfg
2. 即時讀取 UART point cloud
3. 轉成 MARS feature map: (1, 8, 8, 5)
4. 使用 model/MARS.h5 推論 19 joints
5. 顯示 radar point cloud 與 MARS skeleton
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from dataclasses import replace

import numpy as np

try:
	from keras.models import load_model
except ImportError:
	from tensorflow.keras.models import load_model

from pointcloud_pyqtgraph import add_axis_guides
from pointcloud_pyqtgraph import clip_display_points
from pointcloud_pyqtgraph import range_center
from pointcloud_pyqtgraph import range_span
from pointcloud_pyqtgraph import y_to_rgba
from radar_uart import filter_points
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

DEFAULT_CONFIG_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'config_port', default='COM19'))
DEFAULT_DATA_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'data_port', default='COM20'))
DEFAULT_CFG_FILE = str(cfg_get(RADAR_CONFIG, 'radar', 'cfg_file', default='IWRL6844_4T4R_record_high_accuracy.cfg'))
DEFAULT_FRAMES = int(cfg_get(RADAR_CONFIG, 'radar', 'frames', default=-1))
DEFAULT_BAUDRATE_CFG = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_cfg', default=115200))
DEFAULT_BAUDRATE_DATA = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_data', default=1250000))

INTENSITY_MODE = str(cfg_get(RADAR_CONFIG, 'point_output', 'intensity_mode', default='snr_db'))
SNR_NORM_MEAN = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_mean', default=20.0))
SNR_NORM_STD = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_std', default=10.0))
SIDE_INFO_DB_LIMIT = float(cfg_get(RADAR_CONFIG, 'point_output', 'side_info_db_limit', default=100.0))

MAX_POINTS = int(cfg_get(RADAR_CONFIG, 'feature_map', 'max_points', default=64))
FEATURE_DTYPE = str(cfg_get(RADAR_CONFIG, 'feature_map', 'dtype', default='float64'))
TRUNCATE_BEFORE_SORT = bool(cfg_get(RADAR_CONFIG, 'feature_map', 'truncate_before_sort', default=True))

MODEL_FILE = str(cfg_get(RADAR_CONFIG, 'paths', 'model_file', default='MARS.h5'))
PC_X, PC_Y, PC_Z = cfg_range(RADAR_CONFIG, 'point_cloud')
LBL_X, LBL_Y, LBL_Z = cfg_range(RADAR_CONFIG, 'label')


JOINT_NAMES = [
    '脊椎基底', '脊椎中段', '頸部', '頭部',
    '左肩', '左肘', '左手腕',
    '右肩', '右肘', '右手腕',
    '左髖', '左膝', '左踝', '左腳',
    '右髖', '右膝', '右踝', '右腳', '肩胛中心'
]

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (1, 4), (4, 5), (5, 6),
    (1, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16), (16, 17),
    (18, 4), (18, 7), (1, 18),
]

JOINT_ANGLES = {
    'Left elbow': (5, 6, 7),
    'Right elbow': (8, 9, 10),
    'Left knee': (11, 12, 13),
    'Right knee': (15, 16, 17),
}


@dataclass
class RealtimePredictSettings:
	cfg_path: str
	model_path: str
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
	dtype: str = 'float64'
	frame_timeout: float = 0.5
	display_fps: float = 5.0


def frame_to_featuremap(points: np.ndarray, settings: RealtimePredictSettings) -> np.ndarray:
	dtype = np.float64 if settings.dtype == 'float64' else np.float32
	points = filter_points(points, settings.point_filter)
	points = select_mars_points(
		points,
		max_points=settings.max_points,
		truncate_before_sort=settings.truncate_before_sort,
	)

	if points.shape[0] < settings.max_points:
		pad = np.zeros((settings.max_points - points.shape[0], 5), dtype=dtype)
		points = np.vstack((points.astype(dtype, copy=False), pad))
	else:
		points = points.astype(dtype, copy=False)

	return points.reshape(8, 8, 5)


def label_to_joints(label_57: np.ndarray) -> np.ndarray:
	return np.stack([label_57[0:19], label_57[19:38], label_57[38:57]], axis=1)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
	v1 = a - b
	v2 = c - b
	n1 = np.linalg.norm(v1)
	n2 = np.linalg.norm(v2)
	if n1 < 1e-6 or n2 < 1e-6:
		return 0.0
	return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def get_angles(joints: np.ndarray) -> dict[str, float]:
	return {
		name: joint_angle(joints[a], joints[b], joints[c])
		for name, (a, b, c) in JOINT_ANGLES.items()
	}


class RealtimePredictViewer:
	def __init__(self):
		try:
			import pyqtgraph as pg
			import pyqtgraph.opengl as gl
			from pyqtgraph.Qt import QtCore
			from pyqtgraph.Qt import QtGui
			from pyqtgraph.Qt import QtWidgets
		except ImportError as exc:
			raise RuntimeError(
				'需要先安裝 pyqtgraph 3D 依賴：python -m pip install pyqtgraph PyQt5 PyOpenGL'
			) from exc

		self.gl = gl
		self.qt_widgets = QtWidgets
		self.app = QtWidgets.QApplication.instance() or pg.mkQApp('MARS UART Realtime Predict')
		self.window = QtWidgets.QWidget()
		self.window.setWindowTitle('MARS UART Realtime Predict (pyqtgraph)')
		self.window.resize(1320, 760)
		self.window.setStyleSheet(
			'QWidget { background-color: #121418; color: #f4f4f4; }'
			'QLabel { font-family: Consolas, Microsoft JhengHei, sans-serif; }'
		)

		self.pc_title = QtWidgets.QLabel('Radar Point Cloud')
		self.pose_title = QtWidgets.QLabel('MARS Realtime Estimation')
		for label in (self.pc_title, self.pose_title):
			label.setAlignment(QtCore.Qt.AlignCenter)
			label.setStyleSheet('font-size: 18px; font-weight: 600; padding: 8px;')

		self.pc_view = self._make_view(gl, QtGui, PC_X, PC_Y, PC_Z)
		self.pose_view = self._make_view(gl, QtGui, LBL_X, LBL_Y, LBL_Z)

		self.pc_scatter = gl.GLScatterPlotItem(
			pos=np.empty((0, 3), dtype=np.float32),
			color=np.empty((0, 4), dtype=np.float32),
			size=8,
			pxMode=True,
		)
		self.pc_view.addItem(self.pc_scatter)

		self.joint_scatter = gl.GLScatterPlotItem(
			pos=np.empty((0, 3), dtype=np.float32),
			color=(1.0, 0.2, 0.16, 1.0),
			size=9,
			pxMode=True,
		)
		self.pose_view.addItem(self.joint_scatter)
		self.skeleton_lines = gl.GLLinePlotItem(
			pos=np.empty((0, 3), dtype=np.float32),
			color=(1.0, 0.12, 0.08, 1.0),
			width=2.5,
			mode='lines',
			antialias=True,
		)
		self.pose_view.addItem(self.skeleton_lines)

		self.angle_label = QtWidgets.QLabel('')
		self.angle_label.setAlignment(QtCore.Qt.AlignCenter)
		self.angle_label.setMinimumHeight(58)
		self.angle_label.setStyleSheet(
			'font-size: 16px; font-weight: 600; padding: 8px; '
			'border-top: 1px solid rgba(255,255,255,55);'
		)

		left_panel = QtWidgets.QVBoxLayout()
		left_panel.addWidget(self.pc_title)
		left_panel.addWidget(self.pc_view, stretch=1)
		right_panel = QtWidgets.QVBoxLayout()
		right_panel.addWidget(self.pose_title)
		right_panel.addWidget(self.pose_view, stretch=1)

		top_layout = QtWidgets.QHBoxLayout()
		top_layout.addLayout(left_panel, stretch=1)
		top_layout.addLayout(right_panel, stretch=1)

		main_layout = QtWidgets.QVBoxLayout(self.window)
		main_layout.setContentsMargins(8, 8, 8, 8)
		main_layout.addLayout(top_layout, stretch=1)
		main_layout.addWidget(self.angle_label)

		self.window.show()
		self.app.processEvents()

	def _make_view(self, gl, qt_gui, x_range, y_range, z_range):
		display_ranges = (x_range, y_range, z_range)
		x_center = range_center(x_range)
		y_center = range_center(y_range)
		z_center = range_center(z_range)
		x_span = range_span(x_range)
		y_span = range_span(y_range)
		z_span = range_span(z_range)

		view = gl.GLViewWidget()
		view.setBackgroundColor((18, 20, 24))
		view.setCameraPosition(distance=max(x_span, y_span, z_span) * 1.8, elevation=18, azimuth=-62)
		view.opts['center'].setX(x_center)
		view.opts['center'].setY(y_center)
		view.opts['center'].setZ(z_center)

		grid = gl.GLGridItem()
		grid.setSize(x=x_span, y=y_span)
		grid.setSpacing(x=max(x_span / 8.0, 0.25), y=max(y_span / 8.0, 0.25))
		grid.translate(x_center, y_center, z_range[0])
		view.addItem(grid)
		add_axis_guides(view, gl, qt_gui, display_ranges)
		return view

	def update(self, fmap: np.ndarray, pred_57: np.ndarray, frame_number: int):
		joints = label_to_joints(pred_57)
		angles = get_angles(joints)
		points = fmap.reshape(-1, 5)
		points = points[np.any(points != 0, axis=1)]
		points = clip_display_points(points, (PC_X, PC_Y, PC_Z))

		if points.size == 0:
			self.pc_scatter.setData(
				pos=np.empty((0, 3), dtype=np.float32),
				color=np.empty((0, 4), dtype=np.float32),
			)
		else:
			self.pc_scatter.setData(
				pos=points[:, 0:3].astype(np.float32, copy=False),
				color=y_to_rgba(points[:, 1], PC_Y),
				size=8,
				pxMode=True,
			)

		line_points = np.asarray(
			[joints[index] for edge in SKELETON_EDGES for index in edge],
			dtype=np.float32,
		)
		self.joint_scatter.setData(
			pos=joints.astype(np.float32, copy=False),
			color=(1.0, 0.2, 0.16, 1.0),
			size=9,
			pxMode=True,
		)
		self.skeleton_lines.setData(pos=line_points)
		self.pc_title.setText(f'Radar Point Cloud - frame {frame_number}, pts {len(points)}')
		self.angle_label.setText(
			f"Left elbow:  {angles['Left elbow']:5.0f} deg    "
			f"Right elbow: {angles['Right elbow']:5.0f} deg    "
			f"Left knee: {angles['Left knee']:5.0f} deg    "
			f"Right knee: {angles['Right knee']:5.0f} deg"
		)
		self.process_events()

	def process_events(self):
		self.app.processEvents()


def load_mars_model(model_path: str):
	if not os.path.isfile(model_path):
		raise FileNotFoundError(f'找不到模型: {model_path}')
	print(f'[INFO] 載入模型: {model_path}')
	model = load_model(model_path, compile=False)
	expected = tuple(model.input_shape[1:])
	if expected != (8, 8, 5):
		raise ValueError(f'模型 input shape 不是 (8,8,5): {expected}')
	return model


def run_realtime_predict(settings: RealtimePredictSettings) -> int:
	print(f'[INFO] Config 檔案: {settings.cfg_path}')
	print(f'[INFO] Model 檔案: {settings.model_path}')
	print(f'[INFO] Config serial port: {settings.port_cfg}')
	print(f'[INFO] Data serial port: {settings.port_data}')
	print(f'[INFO] intensity_mode: {settings.intensity_mode}')
	print(f'[INFO] filter_roi: {settings.point_filter.roi_enabled}')

	model = load_mars_model(settings.model_path)
	viewer = RealtimePredictViewer()
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

		frame_count = 0
		last_draw_time = 0.0
		draw_interval = 0.0 if settings.display_fps <= 0 else 1.0 / settings.display_fps
		print('[INFO] 開始即時辨識，Ctrl+C 結束。未顯示的 frame 會直接跳過，不做推論。')
		while settings.frames < 0 or frame_count < settings.frames:
			frame = capture.read_frame(timeout_s=settings.frame_timeout)
			if frame is None:
				viewer.process_events()
				continue

			frame_count += 1
			now = time.monotonic()
			if draw_interval != 0.0 and now - last_draw_time < draw_interval:
				continue

			fmap = frame_to_featuremap(frame.points, settings).astype(np.float32, copy=False)
			pred = model.predict(fmap[np.newaxis, ...], batch_size=1, verbose=0)[0]
			viewer.update(fmap, pred, frame.frame_number)
			last_draw_time = now

	except KeyboardInterrupt:
		print('\n[INFO] 使用者中斷。')
	finally:
		if capture is not None:
			capture.close()

	print('[INFO] 結束。')
	return 0


def main() -> int:
	abs_dir = AbsDir()
	default_cfg_path = os.path.join(abs_dir.path_config, DEFAULT_CFG_FILE)
	default_model_path = os.path.join(abs_dir.path_model, MODEL_FILE)

	parser = argparse.ArgumentParser(description='MARS UART 即時姿態辨識')
	parser.add_argument('--cfg', default=default_cfg_path, help='Radar cfg 檔案路徑或檔名')
	parser.add_argument('--model', default=default_model_path, help='MARS .h5 模型路徑')
	parser.add_argument('--port_cfg', default=DEFAULT_CONFIG_PORT, help='Config serial port')
	parser.add_argument('--port_data', default=DEFAULT_DATA_PORT, help='Data serial port')
	parser.add_argument('--frames', type=int, default=-1, help='要辨識的 frame 數，-1 代表持續執行')
	parser.add_argument('--baudrate_cfg', type=int, default=DEFAULT_BAUDRATE_CFG)
	parser.add_argument('--baudrate_data', type=int, default=DEFAULT_BAUDRATE_DATA)
	parser.add_argument('--frame_timeout', type=float, default=0.5)
	parser.add_argument('--display_fps', type=float, default=5.0, help='畫面刷新率；0 代表每個 frame 都更新')
	parser.add_argument('--filter_roi', action='store_true', default=POINT_FILTER_SETTINGS.roi_enabled)
	parser.add_argument('--no_filter_roi', action='store_false', dest='filter_roi')
	args = parser.parse_args()

	cfg_path = resolve_cfg_path(args.cfg)
	model_path = args.model if os.path.isabs(args.model) else os.path.join(abs_dir.path_project_root, args.model)
	if not os.path.isfile(cfg_path):
		print(f'[ERROR] Config 檔案不存在: {cfg_path}')
		return 1
	if not os.path.isfile(model_path):
		print(f'[ERROR] Model 檔案不存在: {model_path}')
		return 1

	settings = RealtimePredictSettings(
		cfg_path=cfg_path,
		model_path=model_path,
		port_cfg=args.port_cfg,
		port_data=args.port_data,
		frames=args.frames,
		baudrate_cfg=args.baudrate_cfg,
		baudrate_data=args.baudrate_data,
		intensity_mode=INTENSITY_MODE,
		snr_norm_mean=SNR_NORM_MEAN,
		snr_norm_std=SNR_NORM_STD,
		side_info_db_limit=SIDE_INFO_DB_LIMIT,
		point_filter=replace(POINT_FILTER_SETTINGS, roi_enabled=args.filter_roi),
		max_points=MAX_POINTS,
		truncate_before_sort=TRUNCATE_BEFORE_SORT,
		dtype=FEATURE_DTYPE,
		frame_timeout=args.frame_timeout,
		display_fps=args.display_fps,
	)
	return run_realtime_predict(settings)


if __name__ == '__main__':
	sys.exit(main())
