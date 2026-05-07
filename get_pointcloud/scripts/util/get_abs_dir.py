import os

def get_abs_dir():
    path_abs = os.path.abspath(__file__)
    dir_abs = os.path.dirname(path_abs)
    path_project_root = os.path.join(dir_abs, '..', '..', '..')
    path_project_root = os.path.normpath(path_project_root)
    
    path_feature = os.path.join(path_project_root, 'feature')
    
    path_pointcloud = os.path.join(path_project_root, 'pointcloud')
    print(f'[INFO] 專案根目錄: {path_project_root}, {"存在 " if os.path.isdir(path_project_root) else "不存在"}')
    print(f'[INFO] Feature 目錄: {path_feature}, {"存在 " if os.path.isdir(path_feature) else "不存在"}')
    print(f'[INFO] PointCloud 目錄: {path_pointcloud}, {"存在 " if os.path.isdir(path_pointcloud) else "不存在"}')

    return path_project_root, path_feature, path_pointcloud
