# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:47:51 2024

@author: wly
"""

import os
from os.path import join, isfile, isdir






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
    def __init__(self, dir_data, data_names=None, mine_format=None):
        self.files = {}
        self.folders = {}

        all_entries = os.listdir(dir_data)
        for entry in all_entries:
            if entry == '__init__.py':
                continue
            if entry == '_class.py':
                continue
            ph_data = join(dir_data, entry)
            
            if isfile(ph_data):
                file_format = entry.split('.')[-1]
                name = '.'.join(entry.split('.')[:-1])

                if mine_format and file_format != mine_format:
                    continue

                if name in self.files:
                    print(f"File '{name}' already exists and will not be overwritten.")
                else:
                    self.files[name] = ph_data
        
        if data_names is not False:
            if data_names is None:
                data_names = get_folders_in_path(dir_data)
            
            for data_name in data_names:
                try:
                    module = __import__(data_name)
                    data_pathsx = getattr(module, 'data_paths', None)
                    
                    if data_name in self.folders:
                        print(f"Folder '{data_name}' already exists and will not be overwritten.")
                    else:
                        self.folders[data_name] = data_pathsx
                except ImportError as e:
                    raise '1'
                    print(f"Error importing {data_name}: {e}")
    def show(self, indent=0):

        prefix = ' ' * indent
        print(f"{prefix}Files:")
        for name, path in list(self.files.items()):
            print(f"{prefix}  {name}: {path}")
        
        print(f"{prefix}Folders:")
        for name, sub_paths in list(self.folders.items()):
            print(f"{prefix}  {name}:")
            if isinstance(sub_paths, Paths):
                sub_paths.show(indent + 4)
            else:
                print(f"{prefix}    (No further structure available)")




os.getcwd()



# class path(str):
#     ...
        


# class paths():
#     def __init__(self,dir_data, data_names=None, mine_format=None):
        
#         all_entries = os.listdir(dir_data)
#         for entry in all_entries:
#             if entry == '__init__.py':
#                 continue
#             ph_data = join(dir_data, entry)
            
            
            
#             if os.path.isfile(ph_data):
#                 try:
#                     file_format = entry.split('.')[-1]
#                     name = '.'.join(entry.split('.')[:-1])
#                 except:
#                     print(ph_data)
#                 if mine_format:
#                     if file_format != mine_format:
#                         continue
#                 if hasattr(self, name):
#                     print(f"Attribute '{name}' already exists and will not be overwritten.")
#                 else :
#                     setattr(self, name, ph_data)
        
#         get_files_in_path(dir_data)
        
#         if data_names is not False:
            
#             if data_names is None:
#                 data_names = get_folders_in_path(dir_data)
            
#             for data_name in data_names:

#                 module = __import__(data_name)
#                 data_pathsx = getattr(module, 'data_paths', None)
                
#                 if hasattr(self, data_name):
#                     print(f"Attribute '{entry}' already exists and will not be overwritten.")
#                 else :
#                     setattr(self, data_name, path(data_pathsx))
#                 # dir_data1 = join(dir_data, data_name)





















