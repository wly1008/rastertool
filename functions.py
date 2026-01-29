# -*- coding: utf-8 -*-

import ast
import os
import numpy as np

import rasterio
from rasterio.enums import Resampling
import geopandas as gpd

from contextlib import  contextmanager
import importlib.metadata as md



def create_raster(**kwargs):
    memfile = rasterio.MemoryFile()
    return memfile.open(**kwargs)

# Create a dataset_opener object to use in several places in this function.
def get_dataset_opener(source):
    if isinstance(source, (str, os.PathLike)):
        return rasterio.open
    else:

        @contextmanager
        def nullcontext(obj):
            try:
                yield obj
            finally:
                pass

        return nullcontext


def read(source,
         indexes=None,
         out_shape=None,
         window=None,
         masked=False,
         resampling=Resampling.nearest,
         boundless=False,
         fill_value=None):
    dataset_opener = get_dataset_opener(source)
    with dataset_opener(source) as src:
        arr = src.read(1, masked=masked)
        profile = src.profile
    
    return arr, profile
    


def readarray(source,
              indexes=None,
              out_shape=None,
              window=None,
              masked=False,
              resampling=Resampling.nearest,
              boundless=False,
              fill_value=None):
    
    dataset_opener = get_dataset_opener(source)
    with dataset_opener(source) as src:
        arr = src.read(1, masked=masked)
    
    return arr

def out(out_path, data, profile, update_stats=False, **profile_update):
    '''
    
    根据 profile 输出 data 至 out_path

    Parameters
    ----------
    out_path : str
        输出路径
    data : array
        数据矩阵.
    profile : dict
        描述栅格数据元数据的字典.
    update_stats : bool, optional
        是否生成或更新统计量. The default is False.
    **profile_update :
        profile 中其他更新参数.

    Returns
    -------
    None.

    '''
    profile.update(profile_update)

    with rasterio.open(out_path, 'w', **profile) as src:
        
        src.write(data)
        if update_stats:
            src.update_stats()  # raserio >= 1.4.0
            # for i in range(1,profile['count']+1):
            #     src.statistics(i)
    





def nan_equal(arr, value):
    """
    判断数组中的值是否等于给定值（支持 NaN 值的比较）。
    
    Parameters
    ----------
    arr : numpy.ndarray
        输入数组。
    value : int, float or None
        要比较的值。
    
    Returns
    -------
    numpy.ndarray
        布尔数组，表示每个元素是否等于给定值。
    """
    arr = np.asarray(arr)
    if value is None or not np.isnan(value):
        return np.equal(arr, value)
    else:
        return np.isnan(arr)





def bounds_to_point(left,bottom,right,top):
    '''界限转点'''
    
    return [[left,top],[left,bottom],[right,bottom],[right,top],[left,top]]

def check_flip(src, n=1):
    '''栅格方法检测与反转'''
    bounds = src.bounds
    if bounds[1] > bounds[3]:
        bounds = [bounds[0],bounds[3],bounds[2],bounds[1]]
        src_arr = np.flip(src.read(),axis=1) 
    else:
        src_arr = src.read()
    if n == 1:
        return src_arr
    elif n == 2:
        return src_arr, bounds

import sys
PACKAGE_ALIAS = {
    # 图像 / 科学计算
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",

    # Web / 网络
    "flask_restful": "flask-restful",
    "flask_sqlalchemy": "flask-sqlalchemy",

    # 数据处理
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",

    # 深度学习
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "tensorflow_core": "tensorflow",

    # 其他常见
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "jwt": "PyJWT",
    "serial": "pyserial",
}
def normalize_package_name(name: str) -> str:
    """将导入名转换为 PyPI 标准包名"""
    return PACKAGE_ALIAS.get(name, name)
def is_builtin_module(module_name):
    """检查是否为内置模块"""
    return module_name in sys.builtin_module_names or module_name in sys.stdlib_module_names
def modules_inspect(PKG_PATH, self=None):
    '''导入模块检测'''
    if self is None:
        self = os.path.basename(PKG_PATH)
    modules = set()
    
    for root, _, files in os.walk(PKG_PATH):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    try:
                        tree = ast.parse(fp.read())
                    except Exception:
                        continue
    
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            if not is_builtin_module(n.name.split(".")[0]):
                                modules.add(n.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and not is_builtin_module(node.module.split(".")[0]):
                            modules.add(node.module.split(".")[0])
    
    print("检测到的依赖模块：")
    modules.discard(self)
    for m in sorted(modules):
        print(" ", m)
    
    print("\n已安装版本：")
    for m in sorted(modules):
        real_name = normalize_package_name(m)
        try:
            print(f"{real_name:20} -> {md.version(real_name)}")
        except Exception:
            print(f"{real_name:20} -> (未找到版本)")


def get_attrs(o, names):
    return [getattr(o, name) for name in names]

def get_geometry(ph_shp, crs=None):
    shp = gpd.read_file(ph_shp) if crs is None else gpd.read_file(ph_shp).to_crs(crs)
    return shp.geometry



if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    modules_inspect(current_dir)
    
    
    
    
    
