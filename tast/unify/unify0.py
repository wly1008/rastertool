# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:39:30 2024

@author: wly
"""

from contextlib import ExitStack
import rasterio
# import mycode.arcmap as ap
import pathlib,os
import numpy as np
from mycode.rio_wrap.core._unify import _unify
from mycode.rio_wrap.core.reproject import reproject
from tqdm import tqdm
from functools import partial,wraps
# __package__ = os.path.dirname(os.path.abspath(__file__))

_temp_dir = os.path.dirname(os.path.abspath(__file__))

def get_attrs(o, names):
    return [getattr(o, name) for name in names]


class noneLock():
    '''空锁'''
    def release(self):...
    def acquire(self):...


# src.res


def unify(raster_in,dst_in=None,out_path=None,
          dst_attrs={'crs':None, 'bounds':None, 'res':None, 'shape':None},
          nodata=None,
          dtype=None,
          get_ds=False,
          Double_operation=False,
          how='nearest',
          crop=True, arr_crop=None,crop_use_index=None,
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
        The default is {'crs':None, 'bounds':None, 'res':None, 'shape':None}.
    mode : str, optional
        裁剪模式，可选round,rio,touch或输入自定义函数，默认为round，详见clip函数
        
    get_ds : bool, optional
        是否获取临时栅格.当out_path为None时有效
        The default is False.
    Double_operation : bool, optional
        是否两次clip操作, 裁剪一次后重采样、重投影再裁剪第二次
        1.减少reproject操作量，原数据远大于目标范围时建议使用
        2.消除分辨率由小变大而外产生的cilp偏移量(？由大变小另行考虑)
        . The default is False.
    how:(str or int) , optional.
    重采样方式，The default is nearest.

    (部分)
    mode:众数，6;
    nearest:临近值，0;
    bilinear:双线性，1;
    cubic_spline:三次卷积，3。
    ...其余见rasterio.enums.Resampling
    
    crop : bool, optional
        是否对目标有效值进行提取
        The default is True.
    arr_crop : array, optional
        有效值掩膜数组，如已输入dst_in请忽略
        The default is None.
    **kwargs : 其他参数
        _temp_dir : str. 中间临时变量位置，默认与本文件相同
        
        fHash : str. 临时变量前缀,
            如多进程中每个进程的循环字符串变量或任意不同变量的哈希值, 防止多进程同时操作一个文件
        
        _temp_ph1 : str 预裁剪输出地址
            _temp_ph1 = karges.get('_temp_ph1', _temp_dir + f'\\{fHash}_clip.tif')
        _temp_ph2 : str.重投影+重采样输出位置
            _temp_ph2 = karges.get('_temp_ph2', _temp_dir + f'\\{fHash}_re.tif')
        
        lock : 进程锁
            多进程使用同一目标栅格dst_in时请输入, 防止多进程同时读取目标栅格
        
    Returns
    -------
    if out_path:生成栅格文件，返回文件地址
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重投影后的栅格矩阵（array）和 profile
    

    '''
    mode='round'  # 暂时固定，很可能不改了
    stacklevel = 3  # 警告级别
    
    _temp_dir = karges.get('_temp_dir',os.path.dirname(os.path.abspath(__file__)))
    
    fHash = karges.get('fHash','')
    
    with ExitStack() as stack:
        
        # 预裁剪
        if Double_operation:
            _temp_ph1 = karges.get('_temp_ph1', _temp_dir + f'\\{fHash}_clip.tif')
            projection = dst_attrs.get('crs',None) or 'geographic'
            
            raster_in = _unify(raster_in, dst_in=dst_in,out_path=_temp_ph1,nodata=nodata,
                       bounds=dst_attrs.get('bounds',None), projection=projection,dtype=dtype,
                       )
        
        src = stack.enter_context(rasterio.open(raster_in)) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
        anames = ['crs', 'bounds', 'res', 'shape']
        # 目标属性
        if dst_in:
            
            lock = karges.get('lock',noneLock())
            
            lock.acquire()  # 进程锁, 防止多进程同步读取相同的dst_in，在锁内读取所有目标属性
            
            
            dst = stack.enter_context(rasterio.open(dst_in)) if issubclass(type(dst_in), (str,pathlib.PurePath)) else dst_in
            
            crs, bounds, res, shape = [getattr(dst, name) for name in anames]
            # size = dst.transform[0]
            
            
            # 裁剪参数
            unify_options = {'ushape':True,'shape':shape}
            if crop:
                arr_crop = dst.read_masks()
            dst_transform = dst.transform
            lock.release()  # 释放
        else:
            crs, bounds, res, shape = [dst_attrs.get(name,None) for name in anames]
        
        
        
        # 原属性
        src_crs, src_bounds, src_res, src_shape = [getattr(src, name) for name in anames]
        # src_size = src.transform[0]
        
        
        # 开始统一变换
        
        # 转投影、重采样
        
        _temp_ph2 = karges.get('_temp_ph2', _temp_dir + f'\\{fHash}_re.tif')
        
        if src_crs == crs:
            # 投影一致
            if src_res == res:
                # 分辨率一致, 裁剪后输出
                
                delete = '!True' if Double_operation else False  # 删除预裁剪生成栅格
                projection = crs or 'geographic'
                return _unify(src, bounds=bounds,
                            out_path=out_path,get_ds=get_ds,
                            mode=mode,
                            projection=projection,
                            crop=crop,
                            arr_crop=arr_crop,
                            crop_use_index=crop_use_index,
                            dst_transform = dst_transform,
                            nodata=nodata,
                            dtype=dtype,
                            stacklevel=stacklevel,
                            unify_options=unify_options,delete=delete)
            
            else:
                # 投影相同, 分辨率不同, 重采样
                ds = reproject(src, crs=None, out_path=_temp_ph2, resolution=res, how=how)
        else:
            # 投影不同(只在相同投影下比较分辨率, 不同投影默认分辨率不同), 重投影+重采样, 
            ds = reproject(src, crs=crs,out_path=_temp_ph2,resolution=res,how=how)
    if Double_operation:
        #删除预裁剪生成栅格
        os.remove(_temp_ph1)
    
    # 裁剪后输出
    delete = True  # 删除转投影、重采样生成的中间栅格
    projection = crs or 'geographic'

    return _unify(ds, bounds=bounds,
                  out_path=out_path,get_ds=get_ds,
                  mode=mode,
                  projection=projection,
                  crop=crop,
                  arr_crop=arr_crop,
                  crop_use_index=crop_use_index,
                  dst_transform=dst_transform,
                  nodata=nodata,
                  dtype=dtype,
                  stacklevel=stacklevel,
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
















