# -*- coding: utf-8 -*-
"""
mars_triplet_demo.py
====================
三欄檢視：
  1. Radar Point Cloud
  2. MARS Prediction
  3. Ground Truth Label

用法：
  python mars_triplet_demo.py
  python mars_triplet_demo.py --input feature/reference/featuremap_test.npy
  python mars_triplet_demo.py --label feature/reference/labels_test.npy
  python mars_triplet_demo.py --model model/MARS.h5
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

from util.AbsDir import AbsDir
from util.radar_config import cfg_get
from util.radar_config import cfg_range
from util.radar_config import load_radar_config
from util.radar_config import resolve_under_root


# ============================================================
# 預設設定
# ============================================================
RADAR_CONFIG = load_radar_config()
DEFAULT_FILE_CLASS = str(cfg_get(RADAR_CONFIG, 'triplet', 'file_class', default='reference'))
DEFAULT_FEATURE_FILE = str(cfg_get(RADAR_CONFIG, 'triplet', 'feature_file', default='featuremap_test.npy'))
DEFAULT_LABEL_FILE = str(cfg_get(RADAR_CONFIG, 'triplet', 'label_file', default='labels_test.npy'))
DEFAULT_MODEL_PATH = str(cfg_get(RADAR_CONFIG, 'paths', 'model_file', default='MARS.h5'))

PC_X, PC_Y, PC_Z = cfg_range(RADAR_CONFIG, 'point_cloud')

JOINT_NAMES = [
    '脊椎基底', '脊椎中段', '頸部', '頭部',
    '左肩', '左肘', '左手腕',
    '右肩', '右肘', '右手腕',
    '左髖', '左膝', '左踝', '左腳',
    '右髖', '右膝', '右踝', '右腳', '肩胛中心'
]

JOINT_ANGLES = {
    'Left elbow':  (4, 5, 6),
    'Right elbow': (7, 8, 9),
    'Left knee':   (11, 12, 13),
    'Right knee':  (15, 16, 17),
}


def label_to_joints(label_57):
    """(57,) -> (19,3)，格式：x×19 | y×19 | z×19"""
    return np.stack([label_57[0:19], label_57[19:38], label_57[38:57]], axis=1)


def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))


def get_angles(j):
    angles = {}
    for name, (a_idx, b_idx, c_idx) in JOINT_ANGLES.items():
        angles[name] = joint_angle(j[a_idx], j[b_idx], j[c_idx])
    return angles


def fmap_to_pts(fmap):
    pts = fmap.reshape(-1, 5)
    return pts[np.any(pts != 0, axis=1)]


def run_inference(fmaps, model_path):
    if not os.path.exists(model_path):
        print(f'[ERROR] 找不到模型: {model_path}')
        sys.exit(1)

    print(f'[推論] 載入模型: {model_path}')
    model = load_model(model_path, compile=False)

    expected = model.input_shape[1:]
    if tuple(fmaps.shape[1:]) != tuple(expected):
        print(f'[ERROR] feature map shape {fmaps.shape[1:]} ≠ 模型期望 {expected}')
        sys.exit(1)

    print(f'[推論] {len(fmaps)} frames...')
    y_pred = model.predict(fmaps, batch_size=256, verbose=1)
    print(f'[推論] 完成 shape={y_pred.shape}')
    return y_pred


def load_npy(path):
    if not os.path.exists(path):
        print(f'[ERROR] 找不到檔案: {path}')
        sys.exit(1)
    return np.load(path).astype(np.float32)


class MARSTripletDemo:
    def __init__(self, fmaps, y_pred, y_true, title_suffix=''):
        self.fmaps = fmaps
        self.pred = y_pred
        self.true = y_true
        self.N = len(fmaps)
        self.idx = 0
        self.title_suffix = title_suffix
        self._build()
        self._update(0)

    def _build(self):
        self.fig = plt.figure(figsize=(15, 6), facecolor='white')
        window_title = 'MARS Triplet Demo'
        if self.title_suffix:
            window_title += f' - {self.title_suffix}'
        self.fig.canvas.manager.set_window_title(window_title)

        outer = gridspec.GridSpec(2, 1, figure=self.fig,
                                  height_ratios=[5, 0.75], hspace=0.08)
        top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.05)
        bot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.6)

        self.ax_pc = self.fig.add_subplot(top[0], projection='3d')
        self.ax_pred = self.fig.add_subplot(top[1], projection='3d')
        self.ax_true = self.fig.add_subplot(top[2], projection='3d')

        ax_txt = self.fig.add_subplot(bot[0])
        ax_sld = self.fig.add_subplot(bot[1])
        ax_txt.axis('off')

        self.angle_text = ax_txt.text(
            0.5, 0.5, '', transform=ax_txt.transAxes,
            ha='center', va='center', fontsize=9, fontfamily='monospace'
        )

        self.slider = Slider(ax_sld, 'Frame', 0, self.N - 1,
                             valinit=0, valstep=1, color='steelblue')
        self.slider.on_changed(lambda v: self._update(int(v)))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _style(self, ax, title):
        ax.set_title(title, fontsize=10, fontweight='bold', pad=3)
        ax.set_xlim(*PC_X)
        ax.set_ylim(*PC_Y)
        ax.set_zlim(*PC_Z)
        ax.set_xlabel('X (m)', fontsize=7, labelpad=1)
        ax.set_ylabel('Y (m)', fontsize=7, labelpad=1)
        ax.set_zlabel('Z (m)', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=15, azim=-60)

    def _update(self, idx):
        idx = int(np.clip(idx, 0, self.N - 1))
        self.idx = idx

        self.ax_pc.cla()
        self.ax_pred.cla()
        self.ax_true.cla()

        # 左欄：點雲
        pts = fmap_to_pts(self.fmaps[idx])
        if len(pts) > 0:
            inten = pts[:, 4].astype(np.float32)
            p1, p99 = np.percentile(inten, [1, 99])
            if p99 - p1 < 1e-6:
                norm_inten = np.zeros_like(inten)
            else:
                norm_inten = np.clip((inten - p1) / (p99 - p1), 0.0, 1.0)

            self.ax_pc.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=norm_inten, cmap='turbo', vmin=0.0, vmax=1.0,
                s=45, alpha=0.95, depthshade=False,
                edgecolors='k', linewidths=0.15
            )
        self._style(self.ax_pc, f'Radar Point Cloud\n{len(pts)} pts')

        # 中欄：推測
        pred_j = label_to_joints(self.pred[idx])
        self.ax_pred.scatter(pred_j[:, 0], pred_j[:, 1], pred_j[:, 2],
                             c='#c0392b', s=35, depthshade=False)
        for ji, name in enumerate(JOINT_NAMES):
            j = pred_j[ji]
            self.ax_pred.text(j[0], j[1], j[2], name, fontsize=7, color='#7f0000')
        self._style(self.ax_pred, 'MARS Prediction')

        # 右欄：label
        true_j = label_to_joints(self.true[idx])
        self.ax_true.scatter(true_j[:, 0], true_j[:, 1], true_j[:, 2],
                             c='#2471a3', s=35, depthshade=False)
        for ji, name in enumerate(JOINT_NAMES):
            j = true_j[ji]
            self.ax_true.text(j[0], j[1], j[2], name, fontsize=7, color='#003f5c')
        self._style(self.ax_true, 'Ground Truth Label')

        # 角度
        ang = get_angles(pred_j)
        self.angle_text.set_text(
            f"Left elbow:  {ang['Left elbow']:5.0f}°    "
            f"Right elbow: {ang['Right elbow']:5.0f}°\n"
            f"Left knee:   {ang['Left knee']:5.0f}°    "
            f"Right knee:  {ang['Right knee']:5.0f}°"
        )

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        step = {'right': 1, 'd': 1, 'left': -1, 'a': -1,
                'pagedown': 50, 'pageup': -50}.get(event.key, 0)
        if step:
            self.slider.set_val(np.clip(self.idx + step, 0, self.N - 1))

    def show(self):
        plt.tight_layout()
        plt.show()


def main():
    abs_dir = AbsDir()

    parser = argparse.ArgumentParser(description='MARS 三欄檢視 demo')
    parser.add_argument('--input', default=None, help='feature map .npy')
    parser.add_argument('--label', default=None, help='label .npy')
    parser.add_argument('--model', default=None, help='模型路徑')
    parser.add_argument('--file_class', default=DEFAULT_FILE_CLASS, help='reference / standard_pose')
    args = parser.parse_args()

    feature_dir = os.path.join(abs_dir.path_feature, args.file_class)

    if args.input is None:
        args.input = os.path.join(feature_dir, DEFAULT_FEATURE_FILE)
    if args.label is None:
        args.label = os.path.join(feature_dir, DEFAULT_LABEL_FILE)
    if args.model is None:
        args.model = os.path.join(abs_dir.path_model, DEFAULT_MODEL_PATH)
    elif not os.path.isabs(args.model):
        args.model = resolve_under_root(args.model)

    print(f'[輸入] {args.input}')
    print(f'[標籤] {args.label}')
    print(f'[模型] {args.model}')

    fmaps = load_npy(args.input)
    y_true = load_npy(args.label)

    if fmaps.ndim != 4 or fmaps.shape[1:] != (8, 8, 5):
        print(f'[ERROR] 需要 (N,8,8,5)，實際 {fmaps.shape}')
        sys.exit(1)

    if y_true.ndim != 2 or y_true.shape[1] != 57:
        print(f'[ERROR] label 需要 (N,57)，實際 {y_true.shape}')
        sys.exit(1)

    y_pred = run_inference(fmaps, args.model)

    file_name = os.path.splitext(os.path.basename(args.input))[0]
    print(f'\n[Demo] {len(y_pred)} frames，← → / A D 換 frame，PageUp/Down 跳 50')
    MARSTripletDemo(fmaps, y_pred, y_true, title_suffix=file_name).show()


if __name__ == '__main__':
    main()
