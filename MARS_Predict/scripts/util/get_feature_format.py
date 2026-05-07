import numpy as np
import sys

if len(sys.argv) < 2:
    print("用法: python inspect_npy_format.py your_file.npy")
    sys.exit(1)

path = sys.argv[1]
arr = np.load(path, allow_pickle=False)

print("檔案:", path)
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("ndim:", arr.ndim)
print("size:", arr.size)

print("\n整體統計:")
print("min:", np.nanmin(arr))
print("max:", np.nanmax(arr))
print("mean:", np.nanmean(arr))
print("std:", np.nanstd(arr))
print("nan count:", np.isnan(arr).sum())
print("zero count:", np.sum(arr == 0))

if arr.ndim == 4:
    n, h, w, c = arr.shape
    print("\n推測格式:")
    print(f"N = {n} frames / samples")
    print(f"H = {h}")
    print(f"W = {w}")
    print(f"C = {c} channels")

    print("\n每個 channel 統計:")
    for ch in range(c):
        data = arr[:, :, :, ch]
        print(
            f"channel {ch}: "
            f"min={np.nanmin(data):.6f}, "
            f"max={np.nanmax(data):.6f}, "
            f"mean={np.nanmean(data):.6f}, "
            f"std={np.nanstd(data):.6f}, "
            f"nonzero={np.count_nonzero(data)}"
        )

    print("\n第 0 個 sample:")
    print(arr[0])

    print("\n第 0 個 sample reshape 成點列表:")
    points = arr[0].reshape(-1, c)
    print(points[:20])

elif arr.ndim == 3:
    print("\n推測可能是:")
    print("(frames, points, features) 或 (samples, h, w)")
    print("\n第 0 筆資料:")
    print(arr[0])

elif arr.ndim == 2:
    print("\n推測可能是:")
    print("(points, features) 或 (samples, features)")
    print("\n前 20 筆:")
    print(arr[:20])

else:
    print("\n無法直接推測，請看 shape 判斷。")