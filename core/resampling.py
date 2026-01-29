# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:39:24 2024

@author: wly
"""



import pathlib
import numpy as np
import numbers
from contextlib import ExitStack

import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

from rastertool.core.reproject import reproject

def create_raster(**kwargs):
    memfile = rasterio.MemoryFile()
    return memfile.open(**kwargs)



def resampling(raster_in, out_path=None, get_ds=True,
               dst_resolution=None, dst_shape=None,
               how='nearest', nodata=None,dtype=None,delete=False,
               num_threads=4, 
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
    dst_resolution: tuple (x resolution, y resolution) or float, optional
        目标分辨率，以目标坐标参考为单位系统.不能与dst_shape一起使用
        
    dst_shape : (dst_height, dst_width) or (count, dst_height, dst_width) tuple or list, optional
        目标行列数。不能与dst_resolution一起使用.
    num_threads : int, optional
        线程数 . Default: 4.
    '''
    # 参数检查
    if dst_shape and dst_resolution:
        raise ValueError("dst_shape和dst_resolution不能一起使用。")
    
    if (dst_shape is None) and (dst_resolution is None):
        raise ValueError("dst_shape和dst_resolution请输入其一。")
    
    # 行列数设置
    if dst_resolution is not None:
        dst_height, dst_width = (None, None)
    elif dst_shape is not None:
        length = len(dst_shape)
        if length == 3:
            dst_height, dst_width = dst_shape[1:]
        elif length == 2:
            dst_height, dst_width  = dst_shape
        else:
            raise ValueError('dst_shape长度错误(%d)，请输入(dst_height, dst_width) 或 (count, dst_height, dst_width)'%length)
        
    # 调用函数
    return reproject(raster_in,
                     dst_in=None,crs=None, # 关闭重投影
                     out_path=out_path, get_ds=get_ds,
                     how=how, dst_nodata=nodata,
                     resolution=dst_resolution, dst_width=dst_width, dst_height=dst_height,
                     num_threads=num_threads,
                     **creation_options)
    
    



def resampling_use_read(raster_in,
                        dst_resolution=None, dst_shape=None,
                        out_path=None, get_ds=True,
                        how='nearest', nodata=None,
                        dtype=None,filled=True,
                        **creation_options):
    
    if not isinstance(how, (str, numbers.Integral)):
        raise TypeError('The how must be a string (str) or an integer (int).')
    
    
    with ExitStack() as stack:
        
        src = stack.enter_context(rasterio.open(raster_in)) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
        
        
        if dst_shape is None and dst_resolution is None:
            
            raise ValueError('please input dst_shape or dst_resolution')
        
        
        # s_transform = src.transform
        # a, b, c, d, e, f, _, _, _ = transform
        bandNames = src.descriptions
        profile = src.profile
        
        if dtype is None:
            dtype = profile['dtype']
        else:
            profile['dtype'] = dtype
        
        
        l, b, r, t = src.bounds
        x,y = src.res
        
        count = src.count
        height, width = src.shape
        
        if dst_shape is None:
            
            if isinstance(dst_resolution,numbers.Number):
                dx = dy = dst_resolution
            else:
                dx, dy = dst_resolution
            
            
            dst_height = int((x * width) / dx)
            dst_width = int((y * height) / dy)
        else:
            length = len(dst_shape)
            if length == 3:
                dst_height, dst_width = dst_shape[1:]
            elif length == 2:
                dst_height, dst_width  = dst_shape
            else:
                raise ValueError('dst_shape长度错误(%d)，请输入(dst_height, dst_width) 或 (count, dst_height, dst_width)'%length)

            
        
        how = getattr(Resampling, how) if isinstance(how, str) else how
        arr = src.read(out_shape=(count, dst_height, dst_width),
                       masked=True,
                       resampling=how, out_dtype=dtype)
        
        transform = from_origin(west=l, north=t, xsize=dx, ysize=dy)
        
        profile.update(
                       transform=transform,
                       width=dst_width,
                       height=dst_height,
                       )
        
        
        try:
            fill_value = np.asarray(nodata, dtype=dtype)
        except (OverflowError, ValueError) as e:
            # Raise TypeError instead of OverflowError or ValueError.
            # OverflowError is seldom used, and the real problem here is
            # that the passed fill_value is not compatible with the ndtype.
            err_msg = "Cannot convert nodata %s to dtype %s"
            raise TypeError(err_msg % (fill_value, dtype)) from e
        if filled:
            arr = arr.filled(nodata)
        else:
            arr.fill_value = nodata
        
        
        # 输出
        if out_path:
            with rasterio.open(out_path, "w", **profile) as out:
                out.write(arr)
                out.descriptions = bandNames

            return out_path
        elif get_ds:
            out = create_raster(**profile)
            out.write(arr)
            out.descriptions = bandNames

            return out
        else:
            return (arr, profile)
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










