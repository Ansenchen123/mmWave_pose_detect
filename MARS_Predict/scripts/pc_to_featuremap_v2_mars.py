# -*- coding: utf-8 -*-
"""
pc_to_featuremap_v2_mars.py
===========================
將 IWR6843 點雲（.mat 或 .npy）轉換成 MARS feature map 格式。

MARS 標準轉換流程：
  1. 依 x→y→z 排序點雲
  2. 截斷或補零到 64 點
  3. reshape 成 (8,8,5)
  4. 座標保留「真實公尺值」（未正規化）

用法：
  # 單個檔案
  python pc_to_featuremap_v2_mars.py --input pointcloud/standard/mars_pointcloud_0506.mat

  # 批量處理整個目錄
  python pc_to_featuremap_v2_mars.py --all_files True

  # 自訂來源目錄
  python pc_to_featuremap_v2_mars.py --input pointcloud/reference/*.mat

輸出位置：
  feature/standard/  （自動建立）
  └─ mars_pointcloud_0506.npy
  └─ ...其他檔案

座標軸定義：
  x = 左右（-1.0 ~ 1.0 m）
  y = 深度（0.3 ~ 3.0 m）
  z = 高度（-1.0 ~ 1.0 m）
"""

import os, sys, argparse
import numpy as np
import glob

import util.get_abs_dir as get_abs_dir

# 預設路徑（可透過 --input 覆蓋）
file_class = 'reference' # 'standard' 或 'reference'
# 路徑由 path_project_root + POINTCLOUD_DIR + POINTCLOUD_FILE 組成
POINTCLOUD_DIR = os.path.normpath(f'pointcloud\{file_class}')
POINTCLOUD_FILE = 'mars_pointcloud_0506_Both_upper_limb_extension'
POINTCLOUD_FILE = f'{POINTCLOUD_FILE if not POINTCLOUD_FILE.endswith(".mat") else os.path.splitext(POINTCLOUD_FILE)[0]}.mat'


# 路徑由 path_project_root + FEATURE_DIR 組成，檔名會與來源的檔名相同
FEATURE_DIR = os.path.normpath(f'feature\{file_class}')



def load_data(path, mat_key='marsData'):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        data = np.load(path).astype(np.float32)
    elif ext == '.mat':
        from scipy.io import loadmat
        mat = loadmat(path)
        candidates = [mat_key, 'marsData', 'radar_data', 'radarData', 'data']
        data = None
        for key in candidates:
            if key in mat:
                data = np.array(mat[key], dtype=np.float32)
                print(f'[載入] key="{key}"  shape={data.shape}')
                break
        if data is None:
            avail = [k for k in mat if not k.startswith('_')]
            print(f'[ERROR] 找不到資料。可用 key: {avail}')
            sys.exit(1)
    else:
        print(f'[ERROR] 不支援格式: {ext}')
        sys.exit(1)
    if data.ndim == 2:
        data = data[np.newaxis]
    print(f'[載入] 完成  shape={data.shape}')
    return data

'''
def frame_to_featuremap(frame, x_range, y_range, grid_h=8, grid_w=8):
    """
    真實公尺值直接投影，不做正規化。
    row → x（左右），col → y（深度）
    同格取 intensity 最高的點，空格補 0。
    """
    fmap       = np.zeros((grid_h, grid_w, 5), dtype=np.float32)
    fmap_inten = np.full((grid_h, grid_w), -1.0)
    x_min, x_max = x_range
    y_min, y_max = y_range

    for pt in frame:
        x, y, z, dop, inten = pt
        if inten <= 1e-6:
            continue
        row = int(np.clip((x - x_min) / (x_max - x_min) * grid_h, 0, grid_h - 1))
        col = int(np.clip((y - y_min) / (y_max - y_min) * grid_w, 0, grid_w - 1))
        if inten > fmap_inten[row, col]:
            fmap[row, col]       = [x, y, z, dop, inten]
            fmap_inten[row, col] = inten
    return fmap
'''
def frame_to_featuremap(frame, max_points=64):
    """
    MARS標準做法：
    1. 不做 spatial binning
    2. 依 x→y→z 排序
    3. 截斷或補零到 64
    4. reshape 成 (8,8,5)
    """

    if frame.shape[0] == 0:
        return np.zeros((8, 8, 5), dtype=np.float32)
    '''
    frame = frame.copy()
    
    frame[:, [0,1]] = frame[:, [1,0]]
    frame[:,2] = -frame[:,2]
    '''
    # ===== ROI filtering =====
    mask = (
        (frame[:,0] > -1.0) & (frame[:,0] < 1.0) &
        (frame[:,1] >  0.3) & (frame[:,1] < 3.0) &
        (frame[:,2] > -1.0) & (frame[:,2] < 1.0)
    )

    frame = frame[mask]
    
    # 🔹 1. sorting（關鍵）
    frame = frame[np.lexsort((frame[:,2], frame[:,1], frame[:,0]))]

    # 🔹 2. 截斷
    if len(frame) > max_points:
        frame = frame[:max_points]

    # 🔹 3. padding
    elif len(frame) < max_points:
        pad = np.zeros((max_points - len(frame), 5), dtype=np.float32)
        frame = np.vstack((frame, pad))

    # 🔹 4. reshape
    fmap = frame.reshape(8, 8, 5)

    return fmap

def convert(pc):
    if pc.ndim == 4:
        print('[INFO] 輸入已是 feature map，直接使用')
        return pc
    N = len(pc)
    print(f'[轉換] {N} frames，座標保留真實公尺（不正規化）')
    fmaps = np.stack([frame_to_featuremap(pc[i]) for i in range(N)])
    used = np.mean(np.any(fmaps != 0, axis=-1))
    pts  = np.array([np.sum(pc[i][:,4] > 1e-6) for i in range(N)])
    print(f'[結果] shape={fmaps.shape}  平均點數={pts.mean():.1f}  grid使用率={used*100:.1f}%')
    if used < 0.3:
        print('       ⚠️  使用率偏低，建議用 pointcloud_enhance.py 做時間累積')
    return fmaps


def main():
    global POINTCLOUD_FILE
    path_project_root, path_feature, path_pointcloud = get_abs_dir.get_abs_dir()

    print(f'[預設] 來源點雲檔案: {os.path.join(POINTCLOUD_DIR, POINTCLOUD_FILE)}')
    print(f'[預設] 輸出特徵圖路徑: {os.path.join(FEATURE_DIR)}')
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default=None)
    # parser.add_argument('--output',  default=None)
    parser.add_argument('--all_files',  default=False, 
                        help='若為True，處理所有在 POINTCLOUD_DIR 路徑下的.mat檔，並保存在同一路徑下的feature資料夾中')
    parser.add_argument('--mat_key', default='marsData')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1.5, 2.0],
                        help='你的雷達 x（左右）範圍，公尺（依你的實際量測範圍設定）')
    parser.add_argument('--y_range', type=float, nargs=2, default=[0.5, 3.0],
                        help='你的雷達 y（深度）範圍，公尺')
    args = parser.parse_args()
    
    if args.all_files:
        POINTCLOUD_FILE = '*.mat'

    if (args.input is None) or (not os.path.isfile(args.input)):
        args.input = os.path.join(path_project_root, POINTCLOUD_DIR, POINTCLOUD_FILE)
    print(f'[輸入] {args.input}')
    input_file = glob.glob(args.input)
    
    if args.all_files:
        for f in input_file:
            print(f'  - {f}')
    
    for input_path in input_file:
        print(f'\n[處理] {input_path}')
        FEATURE_FILE = f'{os.path.splitext(os.path.basename(input_path))[0]}.npy'
        output_file = os.path.join(path_project_root, FEATURE_DIR, FEATURE_FILE)

        pc    = load_data(input_path, args.mat_key)
        fmaps = convert(pc)
        np.save(output_file, fmaps)
        print(f'[儲存] {output_file}')


if __name__ == '__main__':
    main()