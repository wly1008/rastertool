# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:49:30 2024

@author: wly
"""

from contextlib import ExitStack
import rasterio
import mycode.arcmap as ap

import pathlib,os
import numpy as np

from tqdm import tqdm
import multiprocessing
from functools import partial,wraps
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from rasterio.warp import calculate_default_transform

from rasterio.enums import Resampling
from rasterio.warp import reproject as _reproject


def reproject(raster_in, dst_in=None,
              out_path=None, get_ds=True, 
              crs=None,
              how='nearest',
              dst_nodata=None,
              resolution=None, dst_width=None, dst_height=None,
              num_threads=4,
              gcps=None, rpcs=None,
              **creation_options):
    '''
    

    Parameters
    ----------
    raster_in : 地址或栅格类
        输入栅格
    dst_in : 地址或栅格类, optional
        目标栅格
    out_path : str, optional
        输出地址.
        Default: None.
    get_ds : bool, optional
        是否输出临时栅格.当out_path不为None时并不会输出. Default: True.
    crs : CRS, optional
        目标投影,不与dst_in同时使用. Default: None.
    how: (str or int) , optional.
        重采样方式，Default: nearest.

        (部分)\n
        mode:众数，6;\n
        nearest:临近值，0;\n
        bilinear:双线性，1;\n
        cubic_spline:三次卷积，3。\n
        ...其余见rasterio.enums.Resampling
    
    
    dst_nodata : 数字类, optional
        目标无效值，None则与输入栅格相同. Default: None.
    resolution: tuple (x resolution, y resolution) or float, optional
        目标分辨率，以目标坐标参考为单位系统.
        
    dst_width, dst_height: int, optional
        目标行列数。不能与resolution一起使用.
    num_threads : int, optional
        线程数 . Default: 4.
    gcps: sequence of GroundControlPoint, optional
        Ground control points for the source. An error will be raised
        if this parameter is defined together with src_transform or rpcs.
    rpcs: RPC or dict, optional
        Rational polynomial coefficients for the source. An error will
        be raised if this parameter is defined together with src_transform
        or gcps.
    **creation_options :
        目标栅格其他profile更新选项

    Returns
    -------
    if out_path:生成栅格文件，返回文件地址
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重投影后的栅格矩阵（array）和 profile
    
    
    Notes
    ------
    dst_in与crs同时为None时dst_crs==src_crs, 可当作重采样使用

    '''
    
    with ExitStack() as stack:
        
        # 原数据属性获取
        src = stack.enter_context(rasterio.open(raster_in)) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
        
        profile = src.profile
        src_crs = src.crs
        src_nodata = src.nodata
        width, height = src.width, src.height
        bounds = src.bounds
        count = src.count
        
        # 目标投影设置
        if dst_in and crs:
            raise ValueError("目标栅格和目标投影不能一起使用。")
        
        if dst_in:
            dst = stack.enter_context(rasterio.open(dst_in)) if issubclass(type(dst_in), (str,pathlib.PurePath)) else dst_in
            dst_crs = dst.crs
        elif crs:
            dst_crs = crs
        else:
            dst_crs = src_crs
        
        # 目标无效值设置
        if dst_nodata is None:
            dst_nodata = src_nodata
        
        # 计算新的位置信息
        dst_transform, dst_width, dst_height = calculate_default_transform(src_crs, dst_crs, width, height, *bounds,
                                                                           resolution=resolution, dst_width=dst_width, dst_height=dst_height,
                                                                           gcps=gcps, rpcs=rpcs)
        
        # 更新profile
        profile.update({'crs': dst_crs, 'transform': dst_transform, 'width': dst_width, 'height': dst_height,'nodata':dst_nodata})
        
        # 重采样方式
        how = how if isinstance(how, int) else getattr(Resampling, how)
        # 初始化source矩阵
        if 'int8' in str(profile['dtype']):
            # int8时,dst_nodata输入负数无效
            arrn = src.read(out_dtype=np.int16)
            dst_array = np.empty((count, dst_height, dst_width), dtype=np.int16)
        else:
            arrn = src.read()
            dst_array = np.empty((count, dst_height, dst_width), dtype=profile['dtype'])
        # 进行重投影
        _reproject(  
            # 源文件参数
            source=arrn,
            src_crs=src_crs,
            src_nodata=src_nodata,
            src_transform=src.transform,
            # 目标文件参数
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            # 其它配置
            resampling=how,
            num_threads=num_threads)
        if 'int8' in str(profile['dtype']):
            dst_array = dst_array.astype(np.int8)
        
        # 按需求更新
        profile.update(creation_options)
        
        # 输出
        if out_path:
            with rasterio.open(out_path, 'w', **profile) as ds:
                ds.write(dst_array)
            return out_path
        
        elif get_ds:
            ds = ap.create_raster(**profile)
            ds.write(dst_array)
            return ds
        else:
            return dst_array, profile
    
    
    
    
    
    
    
    
    
    




































































