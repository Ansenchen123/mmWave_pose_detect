# -*- coding: utf-8 -*-
"""
mars_predict_demo.py
====================
載入 feature map，用 MARS 預訓練模型推論，顯示互動式姿態估計 demo。

用法：
  # 顯示預設檔案（reference 類別）
  python mars_predict_demo.py

  # --file_class 0|1|2 分別對應 standard|reference|test 類別
  python mars_predict_demo.py --file_class 1

  # 指定自訂 feature map
  python mars_predict_demo.py --input mars_pointcloud_0506.npy --file_class 0

  # 指定模型路徑
  python mars_predict_demo.py --input mars_pointcloud_0506.npy --model MARS.h5

  # 儲存推論結果
  python mars_predict_demo.py --input feature/reference/mars_pointcloud_0506.npy --save_pred pred_output.npy

預設檔案位置：
  feature/reference/featuremap_test.npy

互動控制：
  ← → 鍵     : 逐 frame 切換
  A / D 鍵   : 逐 frame 切換
  PageUp     : 往前跳 50 frames
  PageDown   : 往後跳 50 frames
  滑塊       : 直接選擇 frame

顯示內容（雙視圖）：
  左欄 - Radar Point Cloud:
    • 3D 散點圖
    • x 軸：左右（m）
    • y 軸：深度（m）
    • z 軸：高度（m）
    • 顏色：強度值（turbo colormap）

  右欄 - MARS Estimation (19 joints):
    • 骨骼關節點位置
    • 關節名稱標籤
    • 座標同左欄

  下方 - 關節角度:
    • 左肘角度
    • 右肘角度
    • 左膝角度
    • 右膝角度

輸入格式：
  feature map (.npy)
  形狀：(N, 8, 8, 5)  其中 5 = [x, y, z, doppler, intensity]

輸出格式（可選）：
  預測標籤 (.npy)
  形狀：(N, 57)  即 19 joints × 3 axis
"""

import os, sys, argparse
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
MODEL_FILE = str(cfg_get(RADAR_CONFIG, 'paths', 'model_file', default='MARS.h5'))
BATCH_SIZE = int(cfg_get(RADAR_CONFIG, 'predict', 'batch_size', default=256))


# 軸範圍：與 MARS 原版 demo 完全一致（從論文截圖量測）
PC_X, PC_Y, PC_Z = cfg_range(RADAR_CONFIG, 'point_cloud')
LBL_X, LBL_Y, LBL_Z = cfg_range(RADAR_CONFIG, 'label')

# JOINT_NAMES = [
#     'SpineBase','SpineMid','Neck','Head', 'SpineShoulder',
#     'ShoulderLeft','ElbowLeft','WristLeft',
#     'ShoulderRight','ElbowRight','WristRight',
#     'HipLeft','KneeLeft','AnkleLeft', 'FootLeft',
#     'HipRight','KneeRight','AnkleRight', 'FootRight'
# ]

JOINT_NAMES = [
    'SpineBase','SpineMid','Neck','Head',
    'ShoulderLeft','ElbowLeft','WristLeft',
    'ShoulderRight','ElbowRight','WristRight',
    'HipLeft','KneeLeft','AnkleLeft', 'FootLeft',
    'HipRight','KneeRight','AnkleRight', 'FootRight',
    'SpineShoulder'
]



JOINT_NAMES = [
    '脊椎基底', '脊椎中段', '頸部', '頭部',
    '左肩', '左肘', '左手腕',
    '右肩', '右肘', '右手腕',
    '左髖', '左膝', '左踝', '左腳',
    '右髖', '右膝', '右踝', '右腳', '肩胛中心'
]

# 關節角度對應表
JOINT_ANGLES = {
    'Left elbow':  (5, 6, 7),    # 左肩 → 左肘 → 左手腕
    'Right elbow': (8, 9, 10),   # 右肩 → 右肘 → 右手腕
    'Left knee':   (11, 12, 13), # 左髖 → 左膝 → 左踝
    'Right knee':  (15, 16, 17), # 右髖 → 右膝 → 右踝
}




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


def run_inference(fmaps, model_path='model/MARS.h5', batch_size=256):
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
    y_pred = model.predict(fmaps, batch_size=batch_size, verbose=1)
    print(f'[推論] 完成  shape={y_pred.shape}')
    return y_pred


def label_to_joints(label_57):
    """(57,) → (19,3)，格式：x×19 | y×19 | z×19"""
    return np.stack([label_57[0:19], label_57[19:38], label_57[38:57]], axis=1)


def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1))))


def get_angles(j):
    """計算四個主要關節角度"""
    angles = {}
    for name, (a_idx, b_idx, c_idx) in JOINT_ANGLES.items():
        angles[name] = joint_angle(j[a_idx], j[b_idx], j[c_idx])
    return angles


def fmap_to_pts(fmap):
    pts = fmap.reshape(-1, 5)
    return pts[np.any(pts != 0, axis=1)]


class MARSPredictDemo:
    def __init__(self, fmaps, y_pred, title_suffix=''):
        self.fmaps = fmaps
        self.pred  = y_pred
        self.N     = len(fmaps)
        self.idx   = 0
        self.title_suffix = title_suffix
        self._build()
        self._update(0)

    def _build(self):
        self.fig = plt.figure(figsize=(12, 6), facecolor='white')
        window_title = 'MARS Predict Demo — My IWR6843 Data'
        if self.title_suffix:
            window_title += f' - {self.title_suffix}'
        self.fig.canvas.manager.set_window_title(window_title)
        outer = gridspec.GridSpec(2, 1, figure=self.fig,
                                  height_ratios=[5, 0.75], hspace=0.08)
        top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.05)
        bot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.6)
        self.ax_pc  = self.fig.add_subplot(top[0], projection='3d')
        self.ax_est = self.fig.add_subplot(top[1], projection='3d')
        ax_txt = self.fig.add_subplot(bot[0])
        ax_sld = self.fig.add_subplot(bot[1])
        ax_txt.axis('off')
        self.angle_text = ax_txt.text(
            0.5, 0.5, '', transform=ax_txt.transAxes,
            ha='center', va='center', fontsize=9, fontfamily='monospace')
        self.slider = Slider(ax_sld, 'Frame', 0, self.N-1,
                             valinit=0, valstep=1, color='steelblue')
        self.slider.on_changed(lambda v: self._update(int(v)))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _style(self, ax, title, xlim, ylim, zlim, xl, yl, zl):
        ax.set_title(title, fontsize=10, fontweight='bold', pad=3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel(xl, fontsize=7, labelpad=1)
        ax.set_ylabel(yl, fontsize=7, labelpad=1)
        ax.set_zlabel(zl, fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=15, azim=-60)

    def _update(self, idx):
        idx = int(np.clip(idx, 0, self.N-1))
        self.idx = idx
        self.ax_pc.cla(); self.ax_est.cla()

        # ── 左欄：Radar Point Cloud ───────────────────────────────────────
        pts = fmap_to_pts(self.fmaps[idx])
        if len(pts) > 0:
            inten = pts[:, 4].astype(np.float32)

            # 用百分位數正規化，避免少數極端值把顏色壓扁
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
                s=45,
                alpha=0.95,
                depthshade=False,
                edgecolors='k',
                linewidths=0.15
            )
        self._style(self.ax_pc,
                    f'Radar Point Cloud\n{len(pts)} pts / 64 cells',
                    PC_X, PC_Y, PC_Z,
                    'X left-right (m)', 'Y depth (m)', 'Z height (m)')

        # ── 右欄：MARS Estimation ─────────────────────────────────────────
        joints = label_to_joints(self.pred[idx])
        self.ax_est.scatter(joints[:,0], joints[:,1], joints[:,2],
                            c='#c0392b', s=35, depthshade=False, zorder=5)
        
        for ji, name in enumerate(JOINT_NAMES):
            j = joints[ji]
            self.ax_est.text(
                j[0], j[1], j[2], name,
                fontsize=7, color='#7f0000'
            )
        self._style(self.ax_est,
                    'MARS Estimation (19 joints)',
                    LBL_X, LBL_Y, LBL_Z,
                    'X left-right (m)', 'Y depth (m)', 'Z height (m)')

        # ── 角度 ──────────────────────────────────────────────────────────
        ang = get_angles(joints)
        self.angle_text.set_text(
            f"Left elbow:  {ang['Left elbow']:5.0f}°    "
            f"Right elbow: {ang['Right elbow']:5.0f}°\n"
            f"Left knee:   {ang['Left knee']:5.0f}°    "
            f"Right knee:  {ang['Right knee']:5.0f}°")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        step = {'right':1,'d':1,'left':-1,'a':-1,
                'pagedown':50,'pageup':-50}.get(event.key, 0)
        if step:
            self.slider.set_val(np.clip(self.idx+step, 0, self.N-1))

    def show(self):
        plt.tight_layout()
        plt.show()


def main():
    absDir = AbsDir()
    default_file_class = file_class_from_value(DEFAULT_FILE_CLASS)
    path_feature_dir = absDir.get_feature_dir_by_class(default_file_class)

    global FEATURE_FILE
    FEATURE_FILE = ensure_suffix(FEATURE_FILE, '.npy')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None)
    parser.add_argument('--file_class', default=None, help='檔案類別：test / standard / reference 或 0 / 1 / 2')
    parser.add_argument('--model', default=os.path.join(absDir.path_model, MODEL_FILE))
    parser.add_argument('--save_pred', default=cfg_get(RADAR_CONFIG, 'predict', 'save_pred', default=None))
    parser.add_argument('--auto', default=False)
    args = parser.parse_args()

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
            args.input = FEATURE_FILE
        args.input = resolve_feature_input(absDir, args.file_class, args.input)
        if not os.path.isfile(args.input):
            print(f'[WARNING] 輸入檔案 \"{args.input}\" 不存在')
            os._exit(0)

    fmaps = np.load(args.input).astype(np.float32)
    print(f'[載入] {args.input}  shape={fmaps.shape}')

    if fmaps.ndim != 4 or fmaps.shape[1:] != (8, 8, 5):
        print(f'[ERROR] 需要 (N,8,8,5)，實際 {fmaps.shape}')
        sys.exit(1)

    y_pred = run_inference(fmaps, args.model, batch_size=BATCH_SIZE)

    if args.save_pred:
        np.save(args.save_pred, y_pred)
        print(f'[儲存] {args.save_pred}')

    file_name = os.path.splitext(os.path.basename(args.input))[0]
    print(f'\n[Demo] {len(y_pred)} frames，← → 換 frame，PageUp/Down 跳 50')
    MARSPredictDemo(fmaps, y_pred, title_suffix=file_name).show()


if __name__ == '__main__':
    main()
