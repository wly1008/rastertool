# -*- coding: utf-8 -*-



import ast
import os
import numpy as np

import rasterio
from rasterio.enums import Resampling

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

def out(out_path, data, profile, update_stats=False, **kwargs):
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
    **kwargs :
        profile 中其他更新参数.

    Returns
    -------
    None.

    '''
    profile.update(kwargs)

    with rasterio.open(out_path, 'w', **profile) as src:
        
        src.write(data)
        if update_stats:
            # src.update_stats()  # raserio >= 1.4.0
            for i in range(1,profile['count']+1):
                src.statistics(i)




def copy_raster(source, out_path, update_stats=False, **profile_update):
    '''拷贝栅格，可计算统计量与更新元数据'''
    dataset_opener = get_dataset_opener(source)
    
    with dataset_opener(source) as src:
        data = src.read()
        profile = src.profile
        profile.update(profile_update)
    out(out_path, data, profile, update_stats=update_stats)



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




def renan(source, out_path, ):...







def modules_inspect(PKG_PATH):
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
                            modules.add(n.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            modules.add(node.module.split(".")[0])
    
    print("检测到的依赖模块：")
    for m in sorted(modules):
        print(" ", m)
    
    print("\n已安装版本：")
    for m in sorted(modules):
        try:
            print(f"{m:20} {md.version(m)}")
        except Exception:
            print(f"{m:20} (未找到版本)")


def get_attrs(o, names):
    return [getattr(o, name) for name in names]



