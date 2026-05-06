# -*- coding: utf-8 -*-
"""
mars_predict_demo.py
====================
載入 feature map，用 MARS 預訓練模型推論，顯示互動式姿態估計 demo。

完整工作流程：
  1. 點雲資料 (.mat)
     ↓
  2. pc_to_featuremap_v2_mars.py   → feature map (.npy)
     ↓
  3. mars_predict_demo.py          → 姿態推論 + 視覺化

用法：
  # 顯示預設檔案（reference 類別）
  python mars_predict_demo.py

  # 指定自訂 feature map
  python mars_predict_demo.py --input feature/standard_pose/mars_pointcloud_0506.npy

  # 指定模型路徑
  python mars_predict_demo.py --input feature/reference/*.npy --model model/MARS.h5

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

import util.get_abs_dir as get_abs_dir

# 軸範圍：與 MARS 原版 demo 完全一致（從論文截圖量測）
PC_X  = (-1.0, 1.0)
PC_Y  = ( 0.0, 3.0)
PC_Z  = (-1.0, 1.0)

LBL_X = (-1.0, 1.0)
LBL_Y = ( 0.0, 3.0)
LBL_Z = (-1.0, 1.0)

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
    '脊椎基底','脊椎中段','頸部','頭部','肩胛中心',
    '左肩','左肘','左手腕',
    '右肩','右肘','右手腕',
    '左髖','左膝','左踝','左腳',
    '右髖','右膝','右踝','右腳'
]




def run_inference(fmaps, model_path='model/MARS.h5'):
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
    return {
        'Left elbow':  joint_angle(j[4],  j[5],  j[6]),
        'Right elbow': joint_angle(j[7],  j[8],  j[9]),
        'Left knee':   joint_angle(j[10], j[11], j[12]),
        'Right knee':  joint_angle(j[13], j[14], j[15]),
    }


def fmap_to_pts(fmap):
    pts = fmap.reshape(-1, 5)
    return pts[np.any(pts != 0, axis=1)]


class MARSPredictDemo:
    def __init__(self, fmaps, y_pred):
        self.fmaps = fmaps
        self.pred  = y_pred
        self.N     = len(fmaps)
        self.idx   = 0
        self._build()
        self._update(0)

    def _build(self):
        self.fig = plt.figure(figsize=(12, 6), facecolor='white')
        self.fig.canvas.manager.set_window_title('MARS Predict Demo — My IWR6843 Data')
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
    path_project_root, path_feature, path_pointcloud = get_abs_dir.get_abs_dir()
    file_class = 'reference' # 'standard_pose' 或 'reference'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',     default=None)
    parser.add_argument('--model',     default='model/MARS.h5')
    parser.add_argument('--save_pred', default=None)
    args = parser.parse_args()

    if args.input is None:
        args.input = os.path.join(path_project_root, path_feature, file_class, 'featuremap_test.npy')
    print(f'[輸入] {args.input}, {"存在 " if os.path.isfile(args.input) else "不存在"}')

    fmaps = np.load(args.input).astype(np.float32)
    print(f'[載入] {args.input}  shape={fmaps.shape}')

    if fmaps.ndim != 4 or fmaps.shape[1:] != (8, 8, 5):
        print(f'[ERROR] 需要 (N,8,8,5)，實際 {fmaps.shape}')
        sys.exit(1)

    y_pred = run_inference(fmaps, args.model)

    if args.save_pred:
        np.save(args.save_pred, y_pred)
        print(f'[儲存] {args.save_pred}')

    print(f'\n[Demo] {len(y_pred)} frames，← → 換 frame，PageUp/Down 跳 50')
    MARSPredictDemo(fmaps, y_pred).show()


if __name__ == '__main__':
    main()