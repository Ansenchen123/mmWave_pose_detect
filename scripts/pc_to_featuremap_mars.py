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

from radar_uart import filter_points
from radar_uart import point_filter_settings_from_config
from radar_uart import select_mars_points
from util.AbsDir import AbsDir
from util.AbsDir import FileClass
from util.radar_config import as_bool
from util.radar_config import cfg_get
from util.radar_config import ensure_suffix
from util.radar_config import load_radar_config
from util.radar_config import resolve_under_root

RADAR_CONFIG = load_radar_config()
POINT_FILTER_SETTINGS = point_filter_settings_from_config(RADAR_CONFIG)

DEFAULT_FILE_CLASS = str(cfg_get(RADAR_CONFIG, 'paths', 'default_file_class', default='test'))
DEFAULT_POINTCLOUD_FILE = str(cfg_get(RADAR_CONFIG, 'paths', 'default_pointcloud_file', default='mars_pointcloud_0506_Both_upper_limb_extension.mat'))
DEFAULT_MAT_KEY = str(cfg_get(RADAR_CONFIG, 'conversion', 'mat_key', default='marsData'))
DEFAULT_ALL_FILES = bool(cfg_get(RADAR_CONFIG, 'conversion', 'all_files', default=False))
MAX_POINTS = int(cfg_get(RADAR_CONFIG, 'feature_map', 'max_points', default=64))
FEATURE_DTYPE = str(cfg_get(RADAR_CONFIG, 'feature_map', 'dtype', default='float64'))
TRUNCATE_BEFORE_SORT = bool(cfg_get(RADAR_CONFIG, 'feature_map', 'truncate_before_sort', default=True))


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
def frame_to_featuremap(frame, max_points=64, dtype=np.float64):
    """
    MARS標準做法：
    1. 不做 spatial binning
    2. 依 x→y→z 排序
    3. 截斷或補零到 64
    4. reshape 成 (8,8,5)
    """

    if frame.shape[0] == 0:
        return np.zeros((8, 8, 5), dtype=dtype)

    frame = filter_points(frame.astype(dtype, copy=False), POINT_FILTER_SETTINGS)
    frame = select_mars_points(
        frame,
        max_points=max_points,
        truncate_before_sort=TRUNCATE_BEFORE_SORT,
    )

    # 🔹 padding
    if len(frame) < max_points:
        pad = np.zeros((max_points - len(frame), 5), dtype=dtype)
        frame = np.vstack((frame, pad))

    # 🔹 4. reshape
    fmap = frame.reshape(8, 8, 5)

    return fmap

def convert(pc, max_points=64, dtype=np.float64):
    if pc.ndim == 4:
        print('[INFO] 輸入已是 feature map，直接使用')
        return pc.astype(dtype, copy=False)
    N = len(pc)
    print(f'[轉換] {N} frames，座標保留真實公尺（不正規化）')
    fmaps = np.stack([frame_to_featuremap(pc[i], max_points=max_points, dtype=dtype) for i in range(N)])
    used = np.mean(np.any(fmaps != 0, axis=-1))
    pts  = np.array([np.sum(pc[i][:,4] > 1e-6) for i in range(N)])
    print(f'[結果] shape={fmaps.shape}  平均點數={pts.mean():.1f}  grid使用率={used*100:.1f}%')
    if used < 0.3:
        print('       ⚠️  使用率偏低，建議用 pointcloud_enhance.py 做時間累積')
    return fmaps


def main():
    absDir = AbsDir()
    file_class = file_class_from_value(DEFAULT_FILE_CLASS)
    path_pointcloud_dir = absDir.get_pointcloud_dir_by_class(file_class)
    pointcloud_file_name = ensure_suffix(DEFAULT_POINTCLOUD_FILE, '.mat')
    pointcloud_file = os.path.join(path_pointcloud_dir, pointcloud_file_name)

    feature_file_name = os.path.splitext(pointcloud_file_name)[0] + '.npy'
    path_feature_dir = absDir.get_feature_dir_by_class(file_class)
    feature_file = os.path.join(path_feature_dir, feature_file_name)

    print(f'[預設] 來源點雲檔案: {pointcloud_file}')
    print(f'[預設] 輸出特徵圖路徑: {feature_file}')
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default=None)
    parser.add_argument('--output',  default=None)
    parser.add_argument('--all_files', default=DEFAULT_ALL_FILES, help='True 時處理 file_class 對應 pointcloud 目錄下所有 .mat')
    parser.add_argument('--file_class', default=DEFAULT_FILE_CLASS, help='檔案類別：test / standard / reference 或 0 / 1 / 2')
    parser.add_argument('--mat_key', default=DEFAULT_MAT_KEY)
    args = parser.parse_args()

    file_class = file_class_from_value(args.file_class)
    path_pointcloud_dir = absDir.get_pointcloud_dir_by_class(file_class)
    path_feature_dir = absDir.get_feature_dir_by_class(file_class)
    dtype = np.float64 if FEATURE_DTYPE == 'float64' else np.float32
    
    if as_bool(args.all_files):
        args.input = os.path.join(path_pointcloud_dir, '*.mat')
    elif args.input is None:
        args.input = pointcloud_file
    elif not os.path.isabs(args.input):
        project_candidate = resolve_under_root(args.input)
        if os.path.isfile(project_candidate):
            args.input = project_candidate
        else:
            args.input = os.path.join(path_pointcloud_dir, os.path.basename(args.input))

    print(f'[輸入] {args.input}')
    input_file = glob.glob(args.input)
    
    if as_bool(args.all_files):
        for f in input_file:
            print(f'  - {f}')
    
    for input_path in input_file:
        print(f'\n[處理] {input_path}')
        if args.output is None or as_bool(args.all_files):
            output_file = os.path.join(path_feature_dir, f'{os.path.splitext(os.path.basename(input_path))[0]}.npy')
        elif os.path.isabs(args.output):
            output_file = args.output
        else:
            output_file = resolve_under_root(args.output)

        pc    = load_data(input_path, args.mat_key)
        fmaps = convert(pc, max_points=MAX_POINTS, dtype=dtype)
        np.save(output_file, fmaps)
        print(f'[儲存] {output_file}')


if __name__ == '__main__':
    main()
