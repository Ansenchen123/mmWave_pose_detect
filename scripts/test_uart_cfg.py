from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from radar_uart import filter_points
from radar_uart import point_filter_settings_from_config
from radar_uart import RadarUARTCapture
from radar_uart import RadarUARTSettings
from radar_uart import select_mars_points
from util.radar_config import cfg_get
from util.radar_config import load_radar_config
from util.radar_config import resolve_cfg_path


def test_cfg(cfg_name: str, args, config: dict) -> int:
	cfg_path = resolve_cfg_path(cfg_name)
	if not os.path.isfile(cfg_path):
		print(f'[ERROR] cfg 不存在: {cfg_path}')
		return 1

	point_filter = point_filter_settings_from_config(config)
	max_points = int(cfg_get(config, 'feature_map', 'max_points', default=64))
	truncate_before_sort = bool(cfg_get(config, 'feature_map', 'truncate_before_sort', default=True))

	uart_settings = RadarUARTSettings(
		cfg_path=cfg_path,
		port_cfg=args.port_cfg,
		port_data=args.port_data,
		baudrate_cfg=args.baudrate_cfg,
		baudrate_data=args.baudrate_data,
		intensity_mode=str(cfg_get(config, 'point_output', 'intensity_mode', default='snr_db')),
		snr_norm_mean=float(cfg_get(config, 'point_output', 'snr_norm_mean', default=20.0)),
		snr_norm_std=float(cfg_get(config, 'point_output', 'snr_norm_std', default=10.0)),
		side_info_db_limit=float(cfg_get(config, 'point_output', 'side_info_db_limit', default=100.0)),
	)

	print(f'\n===== 測試 cfg: {os.path.basename(cfg_path)} =====')
	print(f'[INFO] cfg: {cfg_path}')
	print(f'[INFO] cfg port: {args.port_cfg}, data port: {args.port_data}')

	capture = None
	points_per_frame: list[int] = []
	raw_points_per_frame: list[int] = []
	intensities: list[float] = []
	start = time.monotonic()

	try:
		capture = RadarUARTCapture(uart_settings)
		capture.send_config()

		while len(points_per_frame) < args.frames and time.monotonic() - start < args.duration:
			frame = capture.read_frame(timeout_s=args.frame_timeout)
			if frame is None:
				continue

			raw_points = frame.points[np.any(frame.points != 0, axis=1)]
			mars_points = filter_points(frame.points, point_filter)
			mars_points = select_mars_points(
				mars_points,
				max_points=max_points,
				truncate_before_sort=truncate_before_sort,
			)

			raw_points_per_frame.append(len(raw_points))
			points_per_frame.append(len(mars_points))
			if len(mars_points) > 0:
				intensities.extend(mars_points[:, 4].astype(float).tolist())

	except Exception as exc:
		print(f'[ERROR] 測試失敗: {type(exc).__name__}: {exc}')
		return 1
	finally:
		if capture is not None:
			capture.close()

	if not points_per_frame:
		print('[RESULT] 未收到可解析 frame。')
		return 2

	raw_arr = np.array(raw_points_per_frame)
	mars_arr = np.array(points_per_frame)
	print(
		'[RESULT] '
		f'frames={len(points_per_frame)}, '
		f'raw_pts avg/min/max={raw_arr.mean():.1f}/{raw_arr.min()}/{raw_arr.max()}, '
		f'mars_pts avg/min/max={mars_arr.mean():.1f}/{mars_arr.min()}/{mars_arr.max()}'
	)
	if intensities:
		inten = np.array(intensities)
		print(
			'[RESULT] '
			f'intensity min/max/mean={inten.min():.3f}/{inten.max():.3f}/{inten.mean():.3f}'
		)

	return 0


def main() -> int:
	config = load_radar_config()
	parser = argparse.ArgumentParser(description='短測 UART radar cfg 是否能在 COM port 上輸出可解析點雲')
	parser.add_argument('cfg', nargs='*', help='要測試的 cfg 檔名或路徑')
	parser.add_argument('--port_cfg', default=str(cfg_get(config, 'radar', 'config_port', default='COM19')))
	parser.add_argument('--port_data', default=str(cfg_get(config, 'radar', 'data_port', default='COM20')))
	parser.add_argument('--baudrate_cfg', type=int, default=int(cfg_get(config, 'radar', 'baudrate_cfg', default=115200)))
	parser.add_argument('--baudrate_data', type=int, default=int(cfg_get(config, 'radar', 'baudrate_data', default=1250000)))
	parser.add_argument('--frames', type=int, default=20)
	parser.add_argument('--duration', type=float, default=8.0)
	parser.add_argument('--frame_timeout', type=float, default=0.5)
	args = parser.parse_args()

	cfgs = args.cfg or [str(cfg_get(config, 'radar', 'cfg_file', default='IWRL6844_4T4R_record_high_accuracy.cfg'))]
	exit_code = 0
	for cfg in cfgs:
		result = test_cfg(cfg, args, config)
		if result != 0:
			exit_code = result
	return exit_code


if __name__ == '__main__':
	sys.exit(main())
