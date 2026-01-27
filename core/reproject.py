# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:49:30 2024

@author: wly
"""




import pathlib,os
import numpy as np
from contextlib import ExitStack

import rasterio
from rasterio.warp import calculate_default_transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject as _reproject

def cast_value(x, dtype):
    dt = np.dtype(dtype)      # 统一解析字符串 / 类型 / numpy dtype
    return np.array(x, dtype=dt).item()


def create_raster(**kwargs):
    memfile = rasterio.MemoryFile()
    return memfile.open(**kwargs)





def reproject(raster_in, dst_in=None,
              out_path=None, get_ds=True, 
              crs=None,
              how='nearest',
              nodata=None,
              resolution=None, dst_width=None, dst_height=None,
              num_threads=4,
              dtype=None,
              delete=False,

              # gcps=None, rpcs=None,
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
    crs : CRS or str, optional
        目标投影, None且dst_in=None则与输入栅格相同，不与dst_in同时使用. Default: None.
    how: (str or int) , optional.
        重采样方式，Default: nearest.

        (部分)\n
        mode:众数，6;\n
        nearest:临近值，0;\n
        bilinear:双线性，1;\n
        cubic_spline:三次卷积，3。\n
        ...其余见rasterio.enums.Resampling
    
    
    nodata : 数字类, optional
        目标无效值，默认与输入栅格相同(if set), 或者0(GDAL default) . Default: None.
    resolution: tuple (x resolution, y resolution) or float, optional
        目标分辨率，以目标坐标参考为单位系统.
        
    dst_width, dst_height: int, optional
        目标行列数。不能与resolution一起使用.
    num_threads : int, optional
        线程数 . Default: 4.
    dtype : str or numpy dtype opt
        输出栅格值类型,为None时与源栅格一致, 默认为None
    delete : TYPE, optional
        是否删除输入栅格raster_in（清除中间变量）,
        当raster_in为地址时正常执行, 而输入栅格类变量时不会, 除非使用'!True'.
        The default is False.
    **creation_options :
        输出栅格其他profile更新选项

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
        
        # if gcps and rpcs:
        #     raise ValueError("ground control points and rational polynomial",
        #                      " coefficients may not be used together.")
        # 原数据属性获取
        src = stack.enter_context(rasterio.open(raster_in)) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
        # src.read()
        profile = src.profile.copy()
        src_crs = src.crs
        src_nodata = src.nodata
        width, height = src.width, src.height
        bounds = src.bounds
        count = src.count
        bandNames = src.descriptions
        
        # 目标投影设置
        if dst_in and crs:
            raise ValueError("dst_in和crs不能一起使用。")
        
        if dst_in:
            dst = stack.enter_context(rasterio.open(dst_in)) if issubclass(type(dst_in), (str,pathlib.PurePath)) else dst_in
            dst_crs = dst.crs
        elif crs is None:
            dst_crs = src_crs
        else:
            dst_crs = CRS.from_user_input(crs)
        
        # 目标无效值设置
        dst_nodata = nodata
        if dst_nodata is None:
            dst_nodata = src_nodata
        if dst_nodata is None:
            dst_nodata = 0
        
        if dtype is None:
            dtype = profile['dtype']
        else:
            profile['dtype'] = dtype
        
        dst_nodata = cast_value(dst_nodata, dtype)
        
        # 计算新的位置信息
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, width, height, *bounds,
            resolution=resolution, dst_width=dst_width, dst_height=dst_height,
            # gcps=gcps, rpcs=rpcs,
            )
        
        # 更新profile
        profile.update({'crs': dst_crs, 'transform': dst_transform, 'width': dst_width, 'height': dst_height,'nodata':dst_nodata})
        
        # 重采样方式
        how = how if isinstance(how, int) else getattr(Resampling, how)
        # 初始化source矩阵
        if np.dtype(profile['dtype']) == np.int8:
            # int8时,dst_nodata输入负数无效
            arrn = src.read(out_dtype=np.int16)
            dst_array = np.empty((count, dst_height, dst_width), dtype=np.int16)
        else:
            arrn = src.read(out_dtype=dtype)
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
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            dst_transform=dst_transform,
            # 其它配置
            resampling=how,
            num_threads=num_threads,
            )
        if 'int8' in str(profile['dtype']):
            dst_array = dst_array.astype(np.int8)
        # 删除中间栅格
    if delete and issubclass(type(raster_in), (str,pathlib.PurePath)):
        
        os.remove(raster_in)
    # 按需求更新
    profile.update(creation_options)
    profile['descriptions'] = bandNames
    # 输出
    if out_path:
        with rasterio.open(out_path, 'w', **profile) as ds:
            ds.descriptions = bandNames
            ds.write(dst_array)
        return out_path
    
    elif get_ds:
        ds = create_raster(**profile)
        ds.descriptions = bandNames
        ds.write(dst_array)
        return ds
    else:
        return dst_array, profile
    
    



































































