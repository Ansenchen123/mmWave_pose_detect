# -*- coding: utf-8 -*-
"""
my_radar_to_mars.py
===================
把你的 IWR6843 點雲資料轉成 MARS feature map 格式，
餵進 MARS 預訓練模型做 skeleton prediction，並顯示結果。

放到 MARS repo 根目錄執行：
    python my_radar_to_mars.py --input your_radar.mat

你的資料格式 (任一皆可)：
  .mat  → MATLAB 輸出，需指定 key，預設 key = 'radar_data'
  .npy  → numpy array，shape (N, 64, 5) 或 (N, N_pts, 5)

座標欄位順序 (預設)：[x, y, z, doppler, intensity]
  若你的順序不同，請用 --col_order 指定，例如: --col_order 0 2 1 3 4
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ── Keras 相容層 ──────────────────────────────────────────────────────────────
try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

# ─── MARS 19-joint 定義 ──────────────────────────────────────────────────────
JOINT_NAMES = [
    'SpineBase','SpineMid','Neck','Head',
    'ShoulderLeft','ElbowLeft','WristLeft',
    'ShoulderRight','ElbowRight','WristRight',
    'HipLeft','KneeLeft','AnkleLeft',
    'HipRight','KneeRight','AnkleRight',
    'SpineShoulder','HandLeft','HandRight',
]

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1：載入你的雷達資料
# ═══════════════════════════════════════════════════════════════════════════════
def load_my_radar(path, mat_key='radar_data', col_order=None):
    """
    回傳 (N, n_pts, 5) float32，欄位順序 = [x, y, z, doppler, intensity]
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        data = np.load(path).astype(np.float32)
    elif ext in ('.mat', '.mat5'):
        from scipy.io import loadmat
        mat = loadmat(path)
        # 嘗試幾個常見的 key
        candidates = [mat_key, 'radar_data', 'radarData', 'pointcloud',
                      'point_cloud', 'pc', 'data']
        data = None
        for key in candidates:
            if key in mat:
                data = np.array(mat[key], dtype=np.float32)
                print(f'[INFO] 找到 mat key: "{key}"，shape = {data.shape}')
                break
        if data is None:
            print(f'[ERROR] 在 .mat 檔中找不到雷達資料')
            print(f'        可用的 key: {[k for k in mat.keys() if not k.startswith("_")]}')
            sys.exit(1)
    else:
        print(f'[ERROR] 不支援的檔案格式: {ext}')
        sys.exit(1)

    # 修正 shape
    if data.ndim == 2:
        # (N*pts, 5) 或 (pts, 5) → 假設單一 frame
        if data.shape[-1] == 5:
            data = data[np.newaxis]  # (1, pts, 5)
        else:
            print(f'[WARN] 意外的 2D shape: {data.shape}，嘗試 reshape')
    if data.ndim == 4:
        # (N, 8, 8, 5) MARS 格式，直接用
        pass
    elif data.ndim == 3:
        pass  # (N, pts, 5) 正常

    print(f'[INFO] 原始資料 shape: {data.shape}')

    # 重新排欄位
    if col_order is not None:
        data = data[..., col_order]
        print(f'[INFO] 欄位重排為: {col_order} → [x,y,z,doppler,intensity]')

    return data

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2：MARS Feature Map 生成
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_pointcloud(pc_nf, x_range=(-1,1), y_range=(0,3), z_range=(-1,1)):
    """
    將點雲座標正規化到 MARS 訓練資料的空間範圍。
    MARS 資料中人站在雷達前方 0~3m，左右 ±1m，高度 -1~1m。

    pc_nf: (N, pts, 5)
    回傳: (N, pts, 5)，x/y/z 已正規化到 [-1,1] 或 [0,1]
    """
    out = pc_nf.copy()

    # x: 左右
    out[..., 0] = np.clip(
        (pc_nf[..., 0] - x_range[0]) / (x_range[1] - x_range[0]) * 2 - 1,
        -1, 1)
    # y: 深度
    out[..., 1] = np.clip(
        (pc_nf[..., 1] - y_range[0]) / (y_range[1] - y_range[0]),
        0, 1)
    # z: 高度
    out[..., 2] = np.clip(
        (pc_nf[..., 2] - z_range[0]) / (z_range[1] - z_range[0]) * 2 - 1,
        -1, 1)
    # doppler: 正規化到 [-1,1]
    dmax = np.percentile(np.abs(pc_nf[..., 3]), 95) + 1e-6
    out[..., 3] = np.clip(pc_nf[..., 3] / dmax, -1, 1)
    # intensity: 正規化到 [0,1]
    imin = pc_nf[..., 4].min()
    imax = pc_nf[..., 4].max() + 1e-6
    out[..., 4] = (pc_nf[..., 4] - imin) / (imax - imin)

    return out

def pointcloud_to_featuremap(pc_frame, grid_h=8, grid_w=8):
    """
    把單一 frame 的點雲 (n_pts, 5) 投影成 MARS feature map (grid_h, grid_w, 5)。

    MARS 論文的投影邏輯（根據論文 Section 3.2）：
      - 依照 x-y 平面把點雲投影到 8×8 grid
      - 每個 grid cell 取 intensity 最高的點的 [x,y,z,doppler,intensity]
      - 空 cell 補 0

    pc_frame: (n_pts, 5)  [x,y,z,doppler,intensity]，座標需已正規化
    """
    fmap = np.zeros((grid_h, grid_w, 5), dtype=np.float32)
    fmap_intensity = np.full((grid_h, grid_w), -1.0)

    for pt in pc_frame:
        x, y, z, dop, inten = pt

        # 映射到 grid index（用 x 和 y 做 2D 投影）
        # x 正規化到 [-1,1] → grid col [0, grid_w-1]
        col = int((x + 1) / 2 * grid_w)
        col = np.clip(col, 0, grid_w - 1)

        # y 正規化到 [0,1] → grid row [0, grid_h-1]
        row = int(y * grid_h)
        row = np.clip(row, 0, grid_h - 1)

        # 若此 cell 尚空，或此點 intensity 更高，則更新
        if inten > fmap_intensity[row, col]:
            fmap[row, col] = [x, y, z, dop, inten]
            fmap_intensity[row, col] = inten

    return fmap  # (8, 8, 5)

def build_featuremaps(pc_data, grid_h=8, grid_w=8,
                      normalize=True,
                      x_range=(-1,1), y_range=(0,3), z_range=(-1,1)):
    """
    pc_data: (N, n_pts, 5) 或 (N, 8, 8, 5) [已是 feature map]
    回傳:  (N, 8, 8, 5)
    """
    # 若已經是 (N, 8, 8, 5)，直接回傳
    if pc_data.ndim == 4 and pc_data.shape[1:3] == (grid_h, grid_w):
        print(f'[INFO] 輸入已是 feature map 格式 {pc_data.shape}，跳過轉換')
        return pc_data

    N = pc_data.shape[0]

    # 正規化座標
    if normalize:
        pc_data = normalize_pointcloud(pc_data, x_range, y_range, z_range)
        print(f'[INFO] 座標已正規化')

    # 投影每個 frame
    fmaps = np.zeros((N, grid_h, grid_w, 5), dtype=np.float32)
    for i in range(N):
        fmaps[i] = pointcloud_to_featuremap(pc_data[i], grid_h, grid_w)

    print(f'[INFO] Feature map 生成完成: {fmaps.shape}')

    # 統計投影效果
    nonzero = np.mean(np.any(fmaps != 0, axis=-1))
    print(f'[INFO] 平均 grid cell 使用率: {nonzero*100:.1f}%'
          f'（MARS 訓練資料約 40~70%，太低表示點雲稀疏或座標範圍設定有誤）')

    return fmaps

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3：載入 MARS 模型並預測
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference(fmaps, model_path='model/MARS.h5'):
    if not os.path.exists(model_path):
        print(f'[ERROR] 找不到模型: {model_path}')
        sys.exit(1)

    print(f'[INFO] 載入模型: {model_path}')
    model = load_model(model_path, compile=False)
    print(f'       輸入 shape: {model.input_shape}')
    print(f'       輸出 shape: {model.output_shape}')

    # 確認輸入 shape 相符
    expected = model.input_shape[1:]  # (8, 8, 5)
    if fmaps.shape[1:] != expected:
        print(f'[ERROR] Feature map shape {fmaps.shape[1:]} 與模型期望 {expected} 不符')
        sys.exit(1)

    print(f'[INFO] 開始推論 {len(fmaps)} frames...')
    y_pred = model.predict(fmaps, batch_size=256, verbose=1)  # (N, 57)
    print(f'[INFO] 推論完成，輸出 shape: {y_pred.shape}')
    return y_pred

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4：視覺化
# ═══════════════════════════════════════════════════════════════════════════════
def label_to_joints(label_57):
    """(57,) → (19,3)"""
    return np.stack([label_57[0:19], label_57[19:38], label_57[38:57]], axis=1)

def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))

def get_angles(joints):
    return {
        'Left elbow':  joint_angle(joints[4], joints[5], joints[6]),
        'Right elbow': joint_angle(joints[7], joints[8], joints[9]),
        'Left knee':   joint_angle(joints[10], joints[11], joints[12]),
        'Right knee':  joint_angle(joints[13], joints[14], joints[15]),
    }

class MyRadarDemo:
    def __init__(self, fmaps, y_pred, pc_raw=None):
        """
        fmaps : (N, 8, 8, 5)
        y_pred: (N, 57)
        pc_raw: (N, n_pts, 5) 原始點雲（選填，用於左欄更好的視覺化）
        """
        self.fmaps  = fmaps
        self.pred   = y_pred
        self.pc_raw = pc_raw
        self.N      = len(fmaps)
        self.idx    = 0
        self._build()
        self._update(0)

    def _build(self):
        # 有 GT 就三欄，沒有就兩欄
        n_cols = 2
        self.fig = plt.figure(figsize=(5 * n_cols + 2, 6), facecolor='white')
        self.fig.canvas.manager.set_window_title('My Radar → MARS Prediction')

        outer = gridspec.GridSpec(2, 1, figure=self.fig,
                                  height_ratios=[5, 0.75], hspace=0.08)
        top = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[0],
                                               wspace=0.05)
        bot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                               hspace=0.6)

        self.ax_pc  = self.fig.add_subplot(top[0], projection='3d')
        self.ax_est = self.fig.add_subplot(top[1], projection='3d')

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
        ax.set_xlim(-1, 1); ax.set_ylim(0, 3); ax.set_zlim(-1, 1)
        ax.set_xlabel('X (m)', fontsize=7, labelpad=1)
        ax.set_ylabel('Y (m)', fontsize=7, labelpad=1)
        ax.set_zlabel('Z (m)', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=15, azim=-60)

    def _update(self, idx):
        idx = int(np.clip(idx, 0, self.N - 1))
        self.idx = idx
        self.ax_pc.cla(); self.ax_est.cla()

        # ── 左：雷達點雲 ──────────────────────────────────────────────────
        if self.pc_raw is not None:
            pts = self.pc_raw[idx]
        else:
            pts = self.fmaps[idx].reshape(-1, 5)
            pts = pts[np.any(pts != 0, axis=1)]  # 去掉空 cell

        if len(pts) > 0:
            inten = pts[:, 4]
            self.ax_pc.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                               c=inten, cmap='Reds', vmin=0, vmax=1,
                               s=20, depthshade=True)
        self._style(self.ax_pc, 'Radar Point Cloud (IWR6843):')

        # ── 右：MARS 預測 joints ──────────────────────────────────────────
        pred_j = label_to_joints(self.pred[idx])
        self.ax_est.scatter(pred_j[:, 0], pred_j[:, 1], pred_j[:, 2],
                            c='#c0392b', s=30, depthshade=False, zorder=5)
        self._style(self.ax_est, 'MARS Estimation:')

        # ── 角度 ──────────────────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      required=True,
                        help='你的雷達資料路徑 (.npy 或 .mat)')
    parser.add_argument('--mat_key',    default='radar_data',
                        help='.mat 檔中雷達資料的 key（預設: radar_data）')
    parser.add_argument('--col_order',  type=int, nargs=5, default=None,
                        help='欄位順序，例: 0 2 1 3 4 表示把 z 和 y 對調')
    parser.add_argument('--model',      default='model/MARS.h5',
                        help='MARS 模型路徑（預設: model/MARS.h5）')
    parser.add_argument('--no_normalize', action='store_true',
                        help='跳過座標正規化（若資料已是 MARS 空間範圍）')
    parser.add_argument('--x_range',   type=float, nargs=2, default=[-1, 1],
                        help='x 軸範圍 (公尺，預設 -1 1)')
    parser.add_argument('--y_range',   type=float, nargs=2, default=[0, 3],
                        help='y 軸 (深度) 範圍 (公尺，預設 0 3)')
    parser.add_argument('--z_range',   type=float, nargs=2, default=[-1, 1],
                        help='z 軸 (高度) 範圍 (公尺，預設 -1 1)')
    args = parser.parse_args()

    # 1. 載入資料
    pc_raw = load_my_radar(args.input, args.mat_key, args.col_order)

    # 2. 轉成 MARS feature map
    fmaps = build_featuremaps(
        pc_raw,
        normalize=not args.no_normalize,
        x_range=args.x_range,
        y_range=args.y_range,
        z_range=args.z_range,
    )

    # 3. 推論
    y_pred = run_inference(fmaps, args.model)

    # 4. 顯示
    # 傳入原始點雲讓左欄顯示更好
    pc_for_vis = pc_raw if pc_raw.ndim == 3 else None
    demo = MyRadarDemo(fmaps, y_pred, pc_for_vis)
    demo.show()


if __name__ == '__main__':
    main()
