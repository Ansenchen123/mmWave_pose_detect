# -*- coding: utf-8 -*-
"""
mars_run_demo.py
================
放到 MARS repo 根目錄後直接執行：
    python mars_run_demo.py

功能：
  1. 載入 feature/ 裡的 test 資料
  2. 載入或重新訓練模型，做 predict
  3. 跑出和 MARS 論文 Fig.6 一樣的三欄互動 demo：
       Radar Point Cloud | MARS Estimation | Kinect Ground Truth
  4. 顯示四個關節角度 (left/right elbow & knee)
  5. 滑桿 + 鍵盤 ← → 逐 frame 瀏覽

需求：
  pip install tensorflow==2.x keras numpy matplotlib scipy scikit-learn
  (Keras 2.3 / TF 2.2 最佳，但 TF 2.10+ 也能跑，只需忽略 legacy import 警告)
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── 相容新舊版 Keras ─────────────────────────────────────────────────────────
try:
    # Keras 2.x (TF < 2.16)
    from keras.models import load_model, Model
    from keras.layers import (Dense, Input, Flatten, Conv2D, Dropout)
    from keras.optimizers import Adam
    try:
        from keras.layers.normalization import BatchNormalization
    except ImportError:
        from keras.layers import BatchNormalization
except ImportError:
    # Keras 3 / TF >= 2.16
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.layers import (Dense, Input, Flatten, Conv2D,
                                         Dropout, BatchNormalization)
    from tensorflow.keras.optimizers import Adam

import tensorflow as tf

# ─── MARS 19-joint 骨架定義 ──────────────────────────────────────────────────
# label 格式：(frames, 57) = 19 joints × (x19 | y19 | z19)
JOINT_NAMES = [
    'SpineBase','SpineMid','Neck','Head',
    'ShoulderLeft','ElbowLeft','WristLeft',
    'ShoulderRight','ElbowRight','WristRight',
    'HipLeft','KneeLeft','AnkleLeft',
    'HipRight','KneeRight','AnkleRight',
    'SpineShoulder','HandLeft','HandRight',
]

SKELETON_PAIRS = [
    (0,1),(1,16),(16,2),(2,3),
    (16,4),(4,5),(5,6),(6,17),
    (16,7),(7,8),(8,9),(9,18),
    (0,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),
]

# ─── 模型定義（與 MARS_model.py 完全一致）────────────────────────────────────
def define_CNN(in_shape, n_keypoints=57):
    in_one = Input(shape=in_shape)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(in_one)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Dropout(0.4)(x)
    out = Dense(n_keypoints, activation='linear')(x)
    model = Model(in_one, out)
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001, beta_1=0.5),
                  metrics=['mae'])
    return model

# ─── 資料載入 ────────────────────────────────────────────────────────────────
def load_features():
    """載入 MARS feature/ 資料夾裡的 .npy 檔"""
    base = 'feature'
    files = {
        'X_train': os.path.join(base, 'featuremap_train.npy'),
        'X_val':   os.path.join(base, 'featuremap_validate.npy'),
        'X_test':  os.path.join(base, 'featuremap_test.npy'),
        'y_train': os.path.join(base, 'labels_train.npy'),
        'y_val':   os.path.join(base, 'labels_validate.npy'),
        'y_test':  os.path.join(base, 'labels_test.npy'),
    }
    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print(f'[ERROR] 找不到以下檔案，請先 git clone SizheAn/MARS 並確認在 repo 根目錄執行：')
        for k in missing:
            print(f'         {files[k]}')
        sys.exit(1)

    data = {k: np.load(v) for k, v in files.items()}
    print(f'[INFO] 資料載入完成')
    print(f'       X_train : {data["X_train"].shape}')
    print(f'       X_test  : {data["X_test"].shape}')
    print(f'       y_test  : {data["y_test"].shape}')
    return data

# ─── 模型取得（優先載入已存檔，否則快速訓練）────────────────────────────────
def get_model(data, model_path='model/MARS.h5', quick_train=False):
    """
    1. 若 model/MARS.h5 存在 → 直接載入
    2. 否則做 quick_train (5 epochs) 得到可用模型
    """
    if os.path.exists(model_path):
        print(f'[INFO] 載入預訓練模型: {model_path}')
        try:
            model = load_model(model_path, compile=False)
            print(f'[INFO] 模型載入成功')
            return model
        except Exception as e:
            print(f'[WARN] 載入失敗 ({e})，改用快速訓練')

    print('[INFO] 找不到預訓練模型，進行快速訓練 (5 epochs)...')
    print('       (若想完整訓練請執行原版 MARS_model.py)')
    model = define_CNN(data['X_train'][0].shape)
    model.fit(data['X_train'], data['y_train'],
              batch_size=128, epochs=5, verbose=1,
              validation_data=(data['X_val'], data['y_val']))
    os.makedirs('model', exist_ok=True)
    model.save(model_path)
    print(f'[INFO] 模型已存至 {model_path}')
    return model

# ─── label → joints (19,3) ──────────────────────────────────────────────────
def label_to_joints(label_57):
    """(57,) → (19,3)，MARS label 格式：x0..x18 | y0..y18 | z0..z18"""
    j = np.stack([label_57[0:19],
                  label_57[19:38],
                  label_57[38:57]], axis=1)  # (19,3)
    return j

# ─── 關節角度計算 ─────────────────────────────────────────────────────────────
def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1))))

def get_angles(joints):
    return {
        'Left elbow':  joint_angle(joints[4], joints[5], joints[6]),
        'Right elbow': joint_angle(joints[7], joints[8], joints[9]),
        'Left knee':   joint_angle(joints[10], joints[11], joints[12]),
        'Right knee':  joint_angle(joints[13], joints[14], joints[15]),
    }

# ─── 雷達點雲還原（從 feature map 取非零點）──────────────────────────────────
def featuremap_to_pointcloud(fmap):
    """
    fmap: (8,8,5) → 取出有效點 (pts, 5) [x,y,z,doppler,intensity]
    MARS feature map 由 point cloud 做 max-pool 投影，這裡做近似反投影
    """
    pts = fmap.reshape(-1, 5)          # (64, 5)
    mask = np.abs(pts[:, 4]) > 1e-6   # intensity > 0 為有效點
    valid = pts[mask]
    if len(valid) == 0:
        valid = pts                    # fallback 全部顯示
    return valid

# ─── 互動 Demo Viewer ────────────────────────────────────────────────────────
class MARSDemo:
    def __init__(self, X_test, y_pred, y_true):
        self.X    = X_test    # (N, 8, 8, 5)
        self.pred = y_pred    # (N, 57)
        self.true = y_true    # (N, 57)
        self.N    = len(X_test)
        self.idx  = 0
        self._build()
        self._update(0)

    def _build(self):
        self.fig = plt.figure(figsize=(14, 6), facecolor='white')
        self.fig.canvas.manager.set_window_title('MARS Demo')

        outer = gridspec.GridSpec(2, 1, figure=self.fig,
                                  height_ratios=[5, 0.75], hspace=0.08)
        top   = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                                 wspace=0.08)
        bot   = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                                 hspace=0.6)

        self.ax_pc   = self.fig.add_subplot(top[0], projection='3d')
        self.ax_est  = self.fig.add_subplot(top[1], projection='3d')
        self.ax_gt   = self.fig.add_subplot(top[2], projection='3d')

        ax_txt = self.fig.add_subplot(bot[0])
        ax_sld = self.fig.add_subplot(bot[1])
        ax_txt.axis('off')

        self.angle_text = ax_txt.text(
            0.5, 0.5, '', transform=ax_txt.transAxes,
            ha='center', va='center', fontsize=9, fontfamily='monospace')

        self.slider = Slider(ax_sld, 'Frame', 0, self.N - 1,
                             valinit=0, valstep=1, color='steelblue')
        self.slider.on_changed(lambda v: self._update(int(v)))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _style(self, ax, title):
        ax.set_title(title, fontsize=10, fontweight='bold', pad=3)
        lim = 1.0
        ax.set_xlim(-lim, lim); ax.set_ylim(0, 3); ax.set_zlim(-lim, lim)
        ax.set_xlabel('X', fontsize=7, labelpad=1)
        ax.set_ylabel('Y', fontsize=7, labelpad=1)
        ax.set_zlabel('Z', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=15, azim=-60)

    def _draw_skeleton(self, ax, joints, color):
        ax.scatter(joints[:,0], joints[:,1], joints[:,2],
                   c=color, s=22, depthshade=False, zorder=5)

    def _update(self, idx):
        idx = int(np.clip(idx, 0, self.N - 1))
        self.idx = idx

        for ax in [self.ax_pc, self.ax_est, self.ax_gt]:
            ax.cla()

        # ── 左：Radar Point Cloud ──────────────────────────────────────────
        pts = featuremap_to_pointcloud(self.X[idx])   # (pts, 5)
        sc = self.ax_pc.scatter(pts[:,0], pts[:,1], pts[:,2],
                                 c=pts[:,4], cmap='Reds', vmin=0, vmax=1,
                                 s=20, depthshade=True)
        self._style(self.ax_pc, 'Radar Point Cloud:')

        # ── 中：MARS Estimation ───────────────────────────────────────────
        pred_j = label_to_joints(self.pred[idx])
        self._draw_skeleton(self.ax_est, pred_j, '#c0392b')
        self._style(self.ax_est, 'MARS Estimation:')

        # ── 右：Kinect Ground Truth ───────────────────────────────────────
        true_j = label_to_joints(self.true[idx])
        self._draw_skeleton(self.ax_gt, true_j, '#2471a3')
        self._style(self.ax_gt, 'Kinect Ground Truth:')

        # ── 關節角度文字 ───────────────────────────────────────────────────
        ang = get_angles(pred_j)
        self.angle_text.set_text(
            f"Left elbow:  {ang['Left elbow']:5.0f}°    "
            f"Right elbow: {ang['Right elbow']:5.0f}°\n"
            f"Left knee:   {ang['Left knee']:5.0f}°    "
            f"Right knee:  {ang['Right knee']:5.0f}°"
        )

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        step = {'right':1, 'd':1, 'left':-1, 'a':-1,
                'pagedown':50, 'pageup':-50}.get(event.key, 0)
        if step:
            self.slider.set_val(np.clip(self.idx + step, 0, self.N - 1))

    def show(self):
        plt.tight_layout()
        plt.show()

# ─── 主程式 ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 確認在 MARS repo 根目錄
    if not os.path.isdir('feature'):
        print('[ERROR] 請在 MARS repo 根目錄執行此程式')
        print('        cd MARS && python mars_run_demo.py')
        sys.exit(1)

    # 載入資料
    data = load_features()

    # 取得模型（有 .h5 直接載，沒有就快速訓練 5 epochs）
    model = get_model(data)

    # 對 test set 做預測
    print('[INFO] 對 test set 進行預測...')
    y_pred = model.predict(data['X_test'], batch_size=256, verbose=1)
    y_true = data['y_test']

    # 計算 MAE（參考用）
    mae = np.mean(np.abs(y_pred - y_true)) * 100
    print(f'[INFO] Test MAE = {mae:.2f} cm')

    # 啟動互動 Demo
    print(f'[INFO] 共 {len(y_pred)} frames，按 ← → 換 frame，PageUp/Down 跳 50 frames')
    demo = MARSDemo(data['X_test'], y_pred, y_true)
    demo.show()