# -*- coding: utf-8 -*-
"""
show_feature.py
===============
讀入轉換後的 feature map (.npy) 並互動顯示雷達點雲。

用法：
  # 顯示預設檔案（standard_pose 類別）
  python show_feature.py

  # 指定自訂檔案
  python show_feature.py --input feature/reference/mars_pointcloud_0506.npy

  # 快速切換類別
  python show_feature.py --file_class reference

預設檔案位置：
  feature/standard_pose/mars_pointcloud_0506_Both_upper_limb_extension.npy

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

import util.get_abs_dir as get_abs_dir

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 軸範圍
PC_X  = (-1.0, 1.0)
PC_Y  = ( 0.0, 3.0)
PC_Z  = (-1.0, 1.0)


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
                cmap='turbo',
                vmin=0.0, vmax=1.0,
                s=50,
                alpha=0.9,
                depthshade=False,
                edgecolors='k',
                linewidths=0.15
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
    path_project_root, path_feature, path_pointcloud = get_abs_dir.get_abs_dir()
    parser = argparse.ArgumentParser(description='顯示雷達點雲')
    parser.add_argument('--input', default=None,
                        help='輸入 .npy 檔案路徑')
    args = parser.parse_args()

    file_class = 'standard_pose' # 'standard_pose' 或 'reference'
    FEATURE_DIR = os.path.normpath(f'feature\{file_class}')
    FEATURE_FILE = 'mars_pointcloud_0506_Both_upper_limb_extension.npy'
    FEATURE_FILE = f'{FEATURE_FILE if not FEATURE_FILE.endswith(".npy") else os.path.splitext(FEATURE_FILE)[0]}.npy'
    if (args.input is None) or (not os.path.isfile(args.input)):
        args.input = os.path.join(path_project_root, FEATURE_DIR, FEATURE_FILE)
    print(f'[輸入] {args.input}')

    print(f'[載入] {args.input}')
    fmaps = np.load(args.input).astype(np.float32)
    print(f'Shape: {fmaps.shape}')

    if fmaps.ndim != 4 or fmaps.shape[1:] != (8, 8, 5):
        print(f'[WARNING] 期望形狀 (N,8,8,5)，實際 {fmaps.shape}')

    file_name = os.path.splitext(os.path.basename(args.input))[0]
    print(f'\n[控制] ← → 換 frame｜A/D 換 frame｜PageUp/Down 跳 50 frames')
    PointCloudViewer(fmaps, title_suffix=file_name).show()


if __name__ == '__main__':
    main()