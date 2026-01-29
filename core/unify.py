# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:20:01 2024

@author: wly
"""

# TODO 临时文件删除

import os

import warnings
import pathlib


from contextlib import ExitStack
from functools import partial

import rasterio
from rasterio.enums import Resampling

from rastertool.core.clip import clip
from rastertool.core.reproject import reproject
from rastertool.warnings import SetNodataWarning

_temp_dir = os.path.dirname(os.path.abspath(__file__))

def get_attrs(o, names):
    return [getattr(o, name) for name in names]


class noneLock():
    '''空锁'''
    def release(self):...
    def acquire(self):...




def unify(raster_in,dst_in=None,out_path=None,
          dst_attrs={'crs':None, 'bounds':None, 'res':None, 'shape':None},
          nodata=None,
          dtype=None,
          get_ds=False,
          Double=False,
          Triple=False,
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
    Double : bool, optional
        是否两次clip操作, 裁剪一次后重采样、重投影再裁剪第二次
        1.减少reproject操作量，原数据远大于目标范围时建议使用
        2.消除分辨率由小变大而外产生的cilp偏移量(？由大变小另行考虑)
        . The default is False.
    Triple : bool, optional
        是否两次clip操作, 裁剪一次后重投影再裁剪第二次然后重采样, 最后裁剪
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
        keep : 是否保存中间数据
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
        
        # 目标信息获取
        attr_names = ['crs', 'bounds', 'res', 'shape']
        if dst_in is None:
            crs, bounds, res, shape = [dst_attrs.get(name,None) for name in attr_names]
            if crop and arr_crop is None:
                raise ValueError('crop is True, please input dst_in or arr_crop')
        
        else:
            lock = karges.get('lock',noneLock())
            
            lock.acquire()  # 进程锁, 防止多进程同步读取相同的dst_in，在锁内读取所有目标属性
            
            
            dst = stack.enter_context(rasterio.open(dst_in)) if issubclass(type(dst_in), (str,pathlib.PurePath)) else dst_in
            
            crs, bounds, res, shape = [getattr(dst, name) for name in attr_names]
            
            
            # 裁剪参数
            # unify_options = {'ushape':True,'shape':shape}
            if crop:
                arr_crop = dst.read_masks()
            dst_transform = dst.transform
            lock.release()  # 释放

        for name in attr_names[:3]:
            if locals()[name] is None:
                raise ValueError('please input dst_in or %s'%name)
        
        
        
        _clip = partial(clip,
                        # nodata=nodata,
                        mode='round', stacklevel=stacklevel,
                        filled=True, dtype=dtype,
                        with_complement=True,
                        ushape=False, shape=None)
        
        
        # 预裁剪
        keep = karges.get('keep', False)
        if Triple:
            Double = True
        if Double:
            _temp_ph1 = karges.get('_temp_ph1', _temp_dir + f'\\{fHash}_clip.tif')

            raster_in = _clip(raster_in, bounds=bounds, crs=crs, out_path=_temp_ph1, nodata=nodata)
        
        src = stack.enter_context(rasterio.open(raster_in)) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
        
        # 原属性
        src_crs, src_bounds, src_res, src_shape = [getattr(src, name) for name in attr_names]
        
        if nodata is None:
            nodata = src.nodata
        
        if nodata is None:
            nodata = 0
            warnings.warn(
                "源数据无nodata, 自动设定为0, 注意可能产生的影响。 如非所需请为参数nodata赋值",
                SetNodataWarning,
                stacklevel=2
                            )
        _clip = partial(_clip,nodata=nodata)
        
        # 开始统一变换
        
        # 转投影、重采样
        
        _temp_ph2 = karges.get('_temp_ph2', _temp_dir + f'\\{fHash}_repj.tif')
        
        if src_crs == crs:
            # 投影一致
            if src_res == res:
                # 分辨率一致, 裁剪后输出
                
                delete = '!True' if Double else False  # 删除预裁剪生成栅格
                if keep:
                    delete = False
                
                return _clip(src, bounds=bounds,crs=crs,
                             out_path=out_path,get_ds=get_ds,
                             crop=crop,arr_crop=arr_crop,crop_use_index=crop_use_index,
                             delete=delete
                             )
                
                

            
            else:
                # 投影相同, 分辨率不同, 重采样
                ds = reproject(src, crs=None, out_path=_temp_ph2,
                               resolution=res, how=how, dst_nodata=nodata, dtype=dtype)
        else:
            # 投影不同(只在相同投影下比较分辨率, 不同投影默认分辨率不同), 重投影+重采样, 
            if Triple:
                ds_pj = reproject(src, crs=crs,out_path=_temp_ph2,
                                  resolution=None, how=Resampling.nearest, dst_nodata=nodata, dtype=dtype)
                
                delete = True  # 删除转投影、重采样生成的中间栅格
                if keep:
                    delete = False
                
                
                _temp_ph3 = karges.get('_temp_ph3', _temp_dir + f'\\{fHash}_clip2.tif')

                ds_pj_clip = _clip(ds_pj, bounds=bounds, crs=crs, out_path=_temp_ph3,delete=delete)
                
                _temp_ph4 = karges.get('_temp_ph3', _temp_dir + f'\\{fHash}_res.tif')
                ds = reproject(ds_pj_clip, crs=None,out_path=_temp_ph4,
                               resolution=res, how=how, dst_nodata=nodata, dtype=dtype,
                               delete=delete)
            else:
                ds = reproject(src, crs=crs,out_path=_temp_ph2,
                               resolution=res, how=how, dst_nodata=nodata, dtype=dtype)
                
            
    if Double:
        #删除预裁剪生成栅格
        if keep:
            pass
        else:
            os.remove(_temp_ph1)
    
    # 裁剪后输出
    delete = True  # 删除转投影、重采样生成的中间栅格
    if keep:
        delete = False
    return _clip(ds, bounds=bounds,crs=crs,
                 out_path=out_path,get_ds=get_ds,
                 crop=crop,arr_crop=arr_crop,crop_use_index=crop_use_index,
                 delete=delete
                 )
    













