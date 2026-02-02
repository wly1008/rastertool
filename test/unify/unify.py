# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:39:30 2024

@author: wly
"""

from contextlib import ExitStack
import rasterio
# import mycode.arcmap as ap
# import mycode.test.clip_resampling as c1
import pathlib,os
import numpy as np
from mycode.test.clip import clip
from mycode.test.reproject import reproject
from tqdm import tqdm
from functools import partial,wraps
# __package__ = os.path.dirname(os.path.abspath(__file__))

_temp_dir = os.path.dirname(os.path.abspath(__file__))

def get_attrs(o, names):
    return [getattr(o, name) for name in names]


class nonelock():
    def release(self):...
    def acquire(self):...




def unify(raster_in,dst_in=None,out_path=None,
          dst_attrs={'crs':None, 'bounds':None, 'size':None, 'shape':None},
          mode='round',
          nodata='None',
          get_ds=True,
          Double_operation=False,
          how=0,
          crop=True, arr_crop=None,
          **karges):
    '''
    

    Parameters
    ----------
    raster_in : 栅格类或地址
        输入栅格，
    dst_in : 栅格类或地址, optional
        目标栅格.
        The default is None.
    out_path : str, optional
        输出路径. 
        The default is None.
    dst_attrs : dict, optional
        目标属性. 不与dst_in共用
        The default is {'crs':None, 'bounds':None, 'size':None, 'shape':None}.
    mode : str, optional
        裁剪模式，可选round,rio,touch或输入自定义函数，默认为round，详见clip函数
        
    get_ds : bool, optional
        是否获取临时栅格.当out_path为None时有效
        The default is True.
    Double_operation : bool, optional
        是否两次clip操作, 裁剪一次后重采样、重投影再裁剪第二次(原数据远大于目标范围时建议使用)
        . The default is False.
    how : int, optional
        重采样方法.(详见rasterio.enums.Resampling)
        The default is 0(临近值).
    crop : bool, optional
        是否对目标有效值进行提取
        The default is True.
    arr_crop : array, optional
        有效值掩膜数组，如已输入dst_in请忽略
        The default is None.

    Returns
    -------
    if out_path:生成栅格文件，返回文件地址
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重投影后的栅格矩阵（array）和 profile
    

    '''
    

    _temp_dir = karges.get('_temp_dir',os.path.dirname(os.path.abspath(__file__)))
    
    fhsah = karges.get('fhash','')
    
    with ExitStack() as stack:
        
        # 预裁剪
        if Double_operation:
            _temp_ph1 = karges.get('_temp_ph1', _temp_dir + f'\\{fhsah}_clip.tif')
            projection = dst_attrs.get('crs',None) or 'geographic'
            
            raster_in = clip(raster_in, dst_in=dst_in,out_path=_temp_ph1,
                       bounds=dst_attrs.get('bounds',None), projection=projection,
                       )
        
        src = stack.enter_context(rasterio.open(raster_in)) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
        anames = ['crs', 'bounds', 'size', 'shape']
        # 目标属性
        if dst_in:
            
            lock = karges.get('lock',nonelock())
            
            lock.acquire()
            
            
            dst = stack.enter_context(rasterio.open(dst_in)) if issubclass(type(dst_in), (str,pathlib.PurePath)) else dst_in
            
            crs, bounds, shape = [getattr(dst, name) for name in anames if name != 'size']
            size = dst.transform[0]
            
            
            # 裁剪参数
            unify_options = {'ushape':True,'shape':shape}
            if crop:
                arr_crop = dst.read_masks()
            dst_transform = dst.transform
            lock.release()
        else:
            crs, bounds, size, shape = [dst_attrs.get(name,None) for name in anames]
        
        # 裁剪相关
        
        
        
        # 原属性
        src_crs, src_bounds, src_shape = [getattr(src, name) for name in anames if name != 'size']
        src_size = src.transform[0]
        
        
        
        
        # 转投影、重采样
        
        _temp_ph2 = karges.get('_temp_ph2', _temp_dir + f'\\{fhsah}_re.tif')
        delete = True
        if src_crs == crs:
            
            if src_size == size:
                # ds = src
                delete = '!True'
                projection = crs or 'geographic'
                return clip(src, bounds=bounds,
                            out_path=out_path,get_ds=get_ds,
                            mode=mode,
                            projection=projection,
                            crop=crop,
                            arr_crop=arr_crop,
                            dst_transform = dst_transform,
                            nodata=nodata,
                            unify_options=unify_options,delete=delete)
            
            else:
                ds = reproject(src, crs=None, out_path=_temp_ph2, resolution=size, how=how)
        else:
            ds = reproject(src, crs=crs,out_path=_temp_ph2,resolution=size,how=how)
        
    if Double_operation:
        os.remove(_temp_ph1)
    
    
    
    projection = crs or 'geographic'

    return clip(ds, bounds=bounds,
                out_path=out_path,get_ds=get_ds,
                mode=mode,
                projection=projection,
                crop=crop,
                arr_crop=arr_crop,
                dst_transform = dst_transform,
                nodata=nodata,
                unify_options=unify_options,delete=delete)
    
    



# if __name__ == '__main__':
    
    
    # import mycode.arcmap as ap
    
    # raster_in = r'F:/PyCharm/pythonProject1/代码/mycode/测试文件/源数据/1990-5km-tiff.tif'
    # dst_in = r'F:/PyCharm/pythonProject1/代码/mycode/测试文件/源数据/eva_2.tif'
    # out_path = r'F:\PyCharm\pythonProject1\代码\mycode\测试文件\ujh3.tif'
    
    # src = rasterio.open(raster_in)
    # dst = rasterio.open(dst_in)
    
    # # ap.check(src,dst,printf=1)
    # dst.transform
    # # clip(src,dst,out_path,crop=0,nodata=0,)
    # unify(src, dst,out_path,crop=1,nodata=0)
    # ap.unify(src, dst,out_path,crop=1,nodata=0,Extract=1)
















