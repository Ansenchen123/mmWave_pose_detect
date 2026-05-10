import os

def find_default_feature_file(feature_dir):
    """優先找最新的 radar_capture_*.npy，找不到就回退到 feature_dir 內最新的 .npy。"""
    if not os.path.isdir(feature_dir):
        return None

    preferred = sorted(
        [
            os.path.join(feature_dir, name)
            for name in os.listdir(feature_dir)
            if name.startswith('radar_capture_') and name.endswith('.npy')
        ],
        key=os.path.getmtime,
        reverse=True,
    )
    if preferred:
        return preferred[0]

    fallback = sorted(
        [
            os.path.join(feature_dir, name)
            for name in os.listdir(feature_dir)
            if name.endswith('.npy')
        ],
        key=os.path.getmtime,
        reverse=True,
    )
    return fallback[0] if fallback else None