# -*- coding: utf-8 -*-
"""
pc_to_featuremap.py  (v2)
=========================
將你的 IWR6843 點雲轉換成 MARS feature map 格式。

關鍵發現（從 MARS featuremap_test.npy 逆向工程）：
  - MARS featuremap 存的是「真實公尺值」，完全未正規化
  - 座標軸：x=左右, y=深度, z=高度
  - row → x 軸（左右），col → y 軸（深度）

執行：
    python pc_to_featuremap.py --input mars_pointcloud.mat
"""

import os, sys, argparse
import numpy as np


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


def convert(pc, x_range, y_range):
    if pc.ndim == 4:
        print('[INFO] 輸入已是 feature map，直接使用')
        return pc
    N = len(pc)
    print(f'[轉換] {N} frames，座標保留真實公尺（不正規化）')
    fmaps = np.stack([frame_to_featuremap(pc[i], x_range, y_range) for i in range(N)])
    used = np.mean(np.any(fmaps != 0, axis=-1))
    pts  = np.array([np.sum(pc[i][:,4] > 1e-6) for i in range(N)])
    print(f'[結果] shape={fmaps.shape}  平均點數={pts.mean():.1f}  grid使用率={used*100:.1f}%')
    if used < 0.3:
        print('       ⚠️  使用率偏低，建議用 pointcloud_enhance.py 做時間累積')
    return fmaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   required=True)
    parser.add_argument('--mat_key', default='marsData')
    parser.add_argument('--output',  default=None)
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1.5, 2.0],
                        help='你的雷達 x（左右）範圍，公尺（依你的實際量測範圍設定）')
    parser.add_argument('--y_range', type=float, nargs=2, default=[0.5, 3.0],
                        help='你的雷達 y（深度）範圍，公尺')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f'featuremap_{base}.npy'

    pc    = load_data(args.input, args.mat_key)
    fmaps = convert(pc, args.x_range, args.y_range)
    np.save(args.output, fmaps)
    print(f'\n[儲存] {args.output}')
    print(f'[下一步] python mars_predict_demo.py --input {args.output}')


if __name__ == '__main__':
    main()