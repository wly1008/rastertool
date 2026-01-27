# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:03:00 2025

@author: wly
"""

'''Paths还没调试'''

import rasterio
import os
from os.path import join, isfile, isdir
import warnings




class raster():
    rasters = [rasterio.io.DatasetReader,rasterio.io.DatasetWriter,rasterio.io.MemoryFile,rasterio.vrt.WarpedVRT]




class false(Exception):...  # 定义一个特定的错误来跳出函数的循环，便于try捕捉


class TempDir:
    """Context manager to temporarily change the current working directory."""
    
    def __init__(self, new_dir):
        self.new_dir = new_dir
        self.old_dir = None

    def __enter__(self):
        # Store the old directory and change to the new one
        self.old_dir = os.getcwd()
        os.chdir(self.new_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Change back to the old directory, ignoring any exceptions
        os.chdir(self.old_dir)



def get_folders_in_path(path):
    # 确保给定的路径存在且是一个目录
    if os.path.exists(path) and os.path.isdir(path):
        # 列出指定路径下的所有条目
        all_entries = os.listdir(path)
        # 过滤出文件夹
        folders = [entry for entry in all_entries if os.path.isdir(join(path, entry))]
        return folders
    else:
        print(f"路径 {path} 不存在或不是一个有效的目录。")
        return []


def get_files_in_path(path):
    # 确保给定的路径存在且是一个目录
    if os.path.exists(path) and os.path.isdir(path):
        # 列出指定路径下的所有条目
        all_entries = os.listdir(path)
        # 过滤出文件
        files = [entry for entry in all_entries if os.path.isfile(join(path, entry))]
        return files
    else:
        print(f"路径 {path} 不存在或不是一个有效的目录。")
        return []







class Paths:
    '''
    在__init__.py文件中使用
    地址集，获取当前路径下的文件及文件夹（同样在__init__.py中实例化且命名为data_paths）
    当前规范调用代码为：
    
    # 注: 在存储Paths类定义的文件夹(rastertool)的__init__.py中使用调用代码存在重复运行问题，可能造成错误
    from rastertool._Class import Paths
    import os
    
    dir_data = os.path.abspath(os.path.dirname(__file__))
    data_paths = Paths.get_data_paths(dir_data)
    
    '''
    
    @staticmethod
    def get_data_paths(dir_data, data_names=None, mine_format=None):
        '''
        规范调用函数

        Parameters
        ----------
        dir_data : str
            当前文件路径
            在__init__.py中可由以下语句获得:
                dir_data = os.path.abspath(os.path.dirname(__file__))
        data_names : list, optional
            需要的子文件夹名list，为False则不迭代子文件夹，为None则迭代所有子文件夹
            The default is None.
            
        mine_format : 主文件后缀, optional
            如填入则只存储. The default is None.

        Returns
        -------
        data_paths : Paths
            DESCRIPTION.

        '''
        with TempDir(dir_data):
            
            data_paths = Paths(dir_data, data_names=data_names, mine_format=mine_format)

        return data_paths
        
        
    
    def __init__(self, dir_data, data_names=None, mine_format=None, ext=False):
        self.files = {}
        self.folders = {}
        
        rename = []
        
        all_entries = os.listdir(dir_data)
        for entry in all_entries:
            if entry == '__init__.py':
                continue
            if entry == '_class.py':
                continue
            ph_data = join(dir_data, entry)
            
            if isfile(ph_data):
                
                name, file_format = os.path.splitext(entry)
                file_format = file_format.split('.')[-1]
                
                if mine_format and file_format != mine_format:
                    continue

                if name in self.files:
                    # warn_word = f"在路径{dir_data}下，文件名 '{name}' 已存在且此处不覆盖，如需存储同路径同名不同后缀文件请设置ext=True"
                    # warnings.warn(warn_word, UserWarning)
                    
                    if name not in rename:
                        rename.append(name)
                    
                    self.files[entry] = ph_data
                    
                    
                else:
                    self.files[name] = ph_data
        for name in rename:
            ph_data = self.files.pop(name)
            
            self.files[os.path.basename(ph_data)] = ph_data
                
        if data_names is not False:
            if data_names is None:
                data_names = get_folders_in_path(dir_data)
            
            for data_name in data_names:
                if data_name == '__pycache__':
                    continue
                try:
                    module = __import__(data_name)
                    data_pathsx = getattr(module, 'data_paths', None)
                    
                    if data_name in self.folders:
                        print(f"Folder '{data_name}' already exists and will not be overwritten.")
                    else:
                        self.folders[data_name] = data_pathsx
                except ImportError as e:
                    # raise '1'
                    print(f"Error importing {data_name}: {e}")
    def show(self, indent=0):

        prefix = ' ' * indent
        
        ls_files = list(self.files.items())
        if len(ls_files) != 0:
            print(f"{prefix}Files:")
        for name, path in ls_files:
            print(f"{prefix}  {name}: {path}")
        
        ls_folders = list(self.folders.items())
        if len(ls_folders):
            print(f"{prefix}Folders:")
        for name, sub_paths in ls_folders:
            print(f"{prefix}  {name}:")
            if isinstance(sub_paths, Paths):
                sub_paths.show(indent + 4)
            else:
                print(f"{prefix}    (No further structure available)")
                
    def show_tree(self, indent=0, folder_name=None, ext=True):
        prefix = ' ' * indent
        # 显示文件
        for name, path in sorted(self.files.items()):
            
            if ext:
                file_name_with_ext = os.path.basename(path)
                print(prefix + '├─' + file_name_with_ext)
            else:
                print(prefix + '├─' + name)
        
        # 显示子文件夹
        folders = list(self.folders.keys())
        for i, folder in enumerate(sorted(folders)):
            is_last = i == len(folders) - 1
            connector = '└─' if is_last else '├─'
            
            print(prefix + connector + folder)
            if isinstance(self.folders[folder], Paths):
                self.folders[folder].show_tree(indent + 4,
                                               ext=ext)












