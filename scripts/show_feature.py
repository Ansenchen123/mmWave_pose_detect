# -*- coding: utf-8 -*-
"""
show_feature.py
===============
讀入轉換後的 feature map (.npy) 並互動顯示 3D 點雲。

用法：
    # 顯示預設檔案（自動選擇最近一次的 radar_capture_*.npy）
  python show_feature.py

  # 指定自訂檔案
  python show_feature.py --input feature/reference/mars_pointcloud_0506.npy

  # 快速切換類別
  python show_feature.py --file_class reference

預設檔案位置：
    feature/test/radar_capture_*.npy

互動控制：
  ← → 鍵     : 逐 frame 切換
  A / D 鍵   : 逐 frame 切換
  PageUp     : 往前跳 50 frames
  PageDown   : 往後跳 50 frames
  滑塊       : 直接選擇 frame

顯示內容：
  - 3D 點雲散點圖
  - x 軸：左右（m）
  - y 軸：深度（m）
  - z 軸：高度（m）
  - 顏色：強度值（turbo colormap）

輸入格式：
  feature map (.npy)
  形狀：(N, 8, 8, 5)  其中 5 = [x, y, z, doppler, intensity]
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa

from util.AbsDir import AbsDir
from util.AbsDir import FileClass

from util.find_file import find_default_feature_file
from util.radar_config import as_bool
from util.radar_config import cfg_get
from util.radar_config import cfg_range
from util.radar_config import ensure_suffix
from util.radar_config import load_radar_config
from util.radar_config import resolve_under_root

RADAR_CONFIG = load_radar_config()
DEFAULT_FILE_CLASS = str(cfg_get(RADAR_CONFIG, 'paths', 'default_file_class', default='test'))
FEATURE_FILE = str(cfg_get(RADAR_CONFIG, 'paths', 'default_feature_file', default='radar_capture_0.npy'))
SCATTER_CMAP = str(cfg_get(RADAR_CONFIG, 'display', 'scatter', 'cmap', default='turbo'))
SCATTER_SIZE = float(cfg_get(RADAR_CONFIG, 'display', 'scatter', 'size', default=45))
SCATTER_ALPHA = float(cfg_get(RADAR_CONFIG, 'display', 'scatter', 'alpha', default=0.95))
SCATTER_EDGE_COLOR = str(cfg_get(RADAR_CONFIG, 'display', 'scatter', 'edge_color', default='k'))
SCATTER_LINE_WIDTH = float(cfg_get(RADAR_CONFIG, 'display', 'scatter', 'line_width', default=0.15))


plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 軸範圍
PC_X, PC_Y, PC_Z = cfg_range(RADAR_CONFIG, 'point_cloud')


def file_class_from_value(value):
    if isinstance(value, FileClass):
        return value
    text = str(value).strip().lower()
    mapping = {
        'test': FileClass.TEST,
        'standard': FileClass.STANDARD,
        'reference': FileClass.REFERENCE,
        '0': FileClass.TEST,
        '1': FileClass.STANDARD,
        '2': FileClass.REFERENCE,
    }
    return mapping.get(text, FileClass.TEST)


def resolve_feature_input(abs_dir: AbsDir, file_class: FileClass, input_path: str) -> str:
    if os.path.isabs(input_path) and os.path.isfile(input_path):
        return os.path.normpath(input_path)

    project_candidate = resolve_under_root(input_path)
    if os.path.isfile(project_candidate):
        return project_candidate

    return os.path.join(abs_dir.get_feature_dir_by_class(file_class), os.path.basename(input_path))


def fmap_to_pts(fmap):
    """轉換 (8,8,5) feature map 為點雲 (N,5)"""
    pts = fmap.reshape(-1, 5)
    return pts[np.any(pts != 0, axis=1)]



class PointCloudViewer:
    def __init__(self, fmaps, title_suffix=''):
        self.fmaps = fmaps
        self.N = len(fmaps)
        self.idx = 0
        self.title_suffix = title_suffix
        self._build()
        self._update(0)

    def _build(self):
        self.fig = plt.figure(figsize=(10, 8), facecolor='white')
        window_title = 'Point Cloud Viewer'
        if self.title_suffix:
            window_title += f' - {self.title_suffix}'
        self.fig.canvas.manager.set_window_title(window_title)

        outer = gridspec.GridSpec(2, 1, figure=self.fig,
                                  height_ratios=[5, 0.75], hspace=0.08)
        self.ax_pc = self.fig.add_subplot(outer[0], projection='3d')
        ax_sld = self.fig.add_subplot(outer[1])
        
        self.slider = Slider(ax_sld, 'Frame', 0, self.N-1,
                             valinit=0, valstep=1, color='steelblue')
        self.slider.on_changed(lambda v: self._update(int(v)))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _style(self, ax):
        ax.set_title(f'Radar Point Cloud - Frame {self.idx}', 
                     fontsize=11, fontweight='bold', pad=5)
        ax.set_xlim(*PC_X)
        ax.set_ylim(*PC_Y)
        ax.set_zlim(*PC_Z)
        ax.set_xlabel('X left-right (m)', fontsize=9)
        ax.set_ylabel('Y depth (m)', fontsize=9)
        ax.set_zlabel('Z height (m)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.view_init(elev=15, azim=-60)

    def _update(self, idx):
        idx = int(np.clip(idx, 0, self.N-1))
        self.idx = idx
        self.ax_pc.cla()

        pts = fmap_to_pts(self.fmaps[idx])
        if len(pts) > 0:
            inten = pts[:, 4].astype(np.float32)
            
            # 用百分位數正規化
            p1, p99 = np.percentile(inten, [1, 99])
            if p99 - p1 < 1e-6:
                norm_inten = np.zeros_like(inten)
            else:
                norm_inten = np.clip((inten - p1) / (p99 - p1), 0.0, 1.0)

            self.ax_pc.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=norm_inten,
                cmap=SCATTER_CMAP,
                vmin=0.0, vmax=1.0,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                depthshade=False,
                edgecolors=SCATTER_EDGE_COLOR,
                linewidths=SCATTER_LINE_WIDTH
            )
        
        self._style(self.ax_pc)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        step = {'right':1, 'd':1, 'left':-1, 'a':-1,
                'pagedown':50, 'pageup':-50}.get(event.key, 0)
        if step:
            self.slider.set_val(np.clip(self.idx + step, 0, self.N-1))

    def show(self):
        plt.tight_layout()
        plt.show()


def main():    
    parser = argparse.ArgumentParser(description='顯示雷達點雲')
    parser.add_argument('--input', default=None, help='輸入 .npy 檔案名稱')
    parser.add_argument('--auto', default=False, help='自動尋找最新的 radar_capture_*.npy')
    parser.add_argument('--file_class', default=None, help='檔案類別：test / standard / reference 或 0 / 1 / 2')
    args = parser.parse_args()
    
    absDir = AbsDir()
    default_file_class = file_class_from_value(DEFAULT_FILE_CLASS)
    
    args.file_class = default_file_class if args.file_class is None else file_class_from_value(args.file_class)
    print(f'[INFO] 使用的 file_class: {args.file_class}')
    path_feature_dir = absDir.get_feature_dir_by_class(args.file_class)
    
    if as_bool(args.auto):
        args.input = find_default_feature_file(path_feature_dir)
        if args.input is None:
            print(f'[ERROR] 在 \"{path_feature_dir}\" 找不到任何 .npy 檔案')
            os._exit(0)
    else:
        if args.input is None:
            args.input = ensure_suffix(FEATURE_FILE, '.npy')
        args.input = resolve_feature_input(absDir, args.file_class, args.input)
        if not os.path.isfile(args.input):
            print(f'[WARNING] 輸入檔案 \"{args.input}\" 不存在')
            os._exit(0)
            
    print(f'[輸入] {args.input}')
    fmaps = np.load(args.input).astype(np.float32)
    print(f'Shape: {fmaps.shape}')

    if fmaps.ndim != 4 or fmaps.shape[1:] != (8, 8, 5):
        print(f'[WARNING] 期望形狀 (N,8,8,5)，實際 {fmaps.shape}')

    file_name = os.path.splitext(os.path.basename(args.input))[0]
    print(f'\n[控制] ← → 換 frame｜A/D 換 frame｜PageUp/Down 跳 50 frames')
    PointCloudViewer(fmaps, title_suffix=file_name).show()


if __name__ == '__main__':
    main()
