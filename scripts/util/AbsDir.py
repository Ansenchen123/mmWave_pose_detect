import os
from enum import Enum

class FileClass(Enum):
    '''{0: standard, 1: reference, 2: test}'''
    
    STANDARD = 'standard'
    REFERENCE = 'reference'
    TEST = 'test'

    @classmethod
    def from_number(cls, num):
        if isinstance(num, int):
            num = str(num)
        mapping = {
            '0': cls.TEST,
            '1': cls.STANDARD,
            '2': cls.REFERENCE
        }
        return mapping.get(str(num), None)

class AbsDir:
    def __init__(self):
        self.path_abs = os.path.abspath(__file__)
        self.dir_abs = os.path.dirname(self.path_abs)
        self.path_project_root = os.path.join(self.dir_abs, '..', '..')
        self.path_project_root = os.path.normpath(self.path_project_root)
        
        self.path_config = os.path.join(self.path_project_root, 'cfg')
        
        self.path_feature = os.path.join(self.path_project_root, 'feature')
        self.path_feature_test = os.path.join(self.path_feature, 'test')
        self.path_feature_standard = os.path.join(self.path_feature, 'standard')
        self.path_feature_reference = os.path.join(self.path_feature, 'reference')
        
        self.path_pointcloud = os.path.join(self.path_project_root, 'pointcloud')
        self.path_pointcloud_test = os.path.join(self.path_pointcloud, 'test')
        self.path_pointcloud_standard = os.path.join(self.path_pointcloud, 'standard')
        self.path_pointcloud_reference = os.path.join(self.path_pointcloud, 'reference')

        self.path_model = os.path.join(self.path_project_root, 'model')
        
        self.check_all_dir()

    def check_all_dir(self):        
        print(f'[INFO] 專案根目錄: {self.path_project_root}, {"存在 " if os.path.isdir(self.path_project_root) else "不存在"}')
        
        print(f'[INFO] config 目錄: {self.path_config}, {"存在 " if os.path.isdir(self.path_config) else "不存在"}')
        
        print(f'[INFO] feature 目錄: {self.path_feature}, {"存在 " if os.path.isdir(self.path_feature) else "不存在"}')
        print(f'[INFO] feature 目錄: {self.path_feature_reference}, {"存在 " if os.path.isdir(self.path_feature_reference) else "不存在"}')
        print(f'[INFO] feature 目錄: {self.path_feature_standard}, {"存在 " if os.path.isdir(self.path_feature_standard) else "不存在"}')
        print(f'[INFO] feature 目錄: {self.path_feature_test}, {"存在 " if os.path.isdir(self.path_feature_test) else "不存在"}')
        
        print(f'[INFO] PointCloud 目錄: {self.path_pointcloud}, {"存在 " if os.path.isdir(self.path_pointcloud) else "不存在"}')
        print(f'[INFO] PointCloud 目錄: {self.path_pointcloud_reference}, {"存在 " if os.path.isdir(self.path_pointcloud_reference) else "不存在"}')
        print(f'[INFO] PointCloud 目錄: {self.path_pointcloud_standard}, {"存在 " if os.path.isdir(self.path_pointcloud_standard) else "不存在"}')
        print(f'[INFO] PointCloud 目錄: {self.path_pointcloud_test}, {"存在 " if os.path.isdir(self.path_pointcloud_test) else "不存在"}')
        
        print(f'[INFO] Model 目錄: {self.path_model}, {"存在 " if os.path.isdir(self.path_model) else "不存在"}')

    def get_feature_dir_by_class(self, file_class):
        if file_class is FileClass.TEST:
            return self.path_feature_test
        elif file_class is FileClass.STANDARD:
            return self.path_feature_standard
        elif file_class is FileClass.REFERENCE:
            return self.path_feature_reference
        else:
            raise ValueError(f'未知的 file_class: {file_class}')
        
    def get_pointcloud_dir_by_class(self, file_class):
        if file_class is FileClass.TEST:
            return self.path_pointcloud_test
        elif file_class is FileClass.STANDARD:
            return self.path_pointcloud_standard
        elif file_class is FileClass.REFERENCE:
            return self.path_pointcloud_reference
        else:
            raise ValueError(f'未知的 file_class: {file_class}')

def main():
    abs_dir = AbsDir()
    abs_dir.check_all_dir()

    print(abs_dir.path_pointcloud)
    print(abs_dir.path_pointcloud_test)
    print(abs_dir.path_pointcloud_standard)
    print(abs_dir.path_pointcloud_reference)

if __name__ == "__main__":
    main()