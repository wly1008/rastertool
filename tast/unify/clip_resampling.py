# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:10:57 2023

@author: wly
"""

from rasterio.warp import calculate_default_transform
# from mycode.arcmap import *
from rasterio.enums import Resampling
import rasterio
import pathlib
from rasterio.windows import Window
import mycode.arcmap as ap
import numpy as np
import warnings
from tqdm import tqdm
import rasterio.merge


def create_raster(**kwargs):
    '''
    在内存中创建栅格数据不写出

    Parameters
    ----------
    **kwargs : TYPE
        profile

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    memfile = rasterio.MemoryFile()
    return memfile.open(**kwargs)


def window(raster_in, shape=None, size=None, initial_offset=None):
    '''
    

    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        栅格数据或栅格地址
    shape : tuple
          (height,width)
        分割为 height*width个窗口
    size : int、float or tuple
          (ysize,xsize)
        窗口的尺寸大小，多余的会生成独立的小窗口不会并入前一个窗口
    initial_offset : tuple
                    (initial_offset_x, initial_offset_y)
        初始偏移量,默认为(0,0)

    Returns
    -------
    windows : TYPE
        窗口集
    inxs : TYPE
        对应窗口在栅格中的位置索引

    '''

    assert shape or size, '请填入shape or size'
    assert not (shape and size), 'shape 与 size只填其中一个'
    
    src = rasterio.open(raster_in) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
    if shape:
        xsize, xend0 = divmod(src.width, shape[1])
        ysize, yend0 = divmod(src.height, shape[0])
        xend = xsize + xend0
        yend = ysize + yend0
    else:
        if isinstance(size, (int ,float)):
            xsize = size
            ysize = size
        else:
            ysize, xsize = size
        xend = src.width % xsize or xsize
        yend = src.height % ysize or ysize
        
        
        s0 = int(np.ceil(src.height / ysize))
        s1 = int(np.ceil(src.width / xsize))
        shape = (s0, s1)
        
    
    initial_offset_x, initial_offset_y = initial_offset or (0,0)
    
    y_off = initial_offset_y
    inxs = []
    # inx = {}
    windows = []
    for y_inx,ax0 in enumerate(range(shape[0])):
        
        x_off = initial_offset_x
        height = yend if ax0 == (shape[0] - 1) else ysize
        for x_inx,ax1 in enumerate(range(shape[1])):

            width = xend if ax1 == (shape[1] - 1) else xsize
            windows.append(Window(x_off, y_off, width, height))
            
            '''
            
            start = x_off
            end = x_off + width
            inx['x'] = (start, end)

            start = y_off
            end = y_off + height
            inx['y'] = (start, end)

            inxs.append(inx.copy())
            '''
            inxs.append((y_inx,x_inx))
            
            x_off += width
        
        y_off += height

    return windows, inxs



# def clip(raster_in,
#          dst_in=None, bounds=None,
#          inner=False,
#          Extract=False, mask=False,
#          out_path=None, get_ds=True):




def clip(raster_in,
         dst_in=None,bounds=None,
         mode="",
         inner=False, union=False,
         out_path=None, get_ds=True,
         Extract=False,
         win_shape=None,win_size=None,
         Tqbm=True,
         unify=False,
         u=False):
    '''
    

    Parameters
    ----------
    raster_in : str
        被裁剪的栅格数据或栅格地址
    dst_in : str, optional
        目标范围的栅格数据或栅格地址
    bounds : tuple, optional
        目标范围,(left, bottom, right, top),bounds与dst_in只需填其中一个，如都填的话bounds优先级更高dst_in失效
    win_shape : tuple, optional
        (height,width)
        分窗口运行，如数据量过大可使用此参数，自己试试分成 几乘几 个窗口可以运行.e.g.(3,3)-->分成九份写出
    out_path : str, optional
        输出路径，如不填返回栅格矩阵和栅格profile
    Tqbm : bool, optional
        是否显示进度条，The default is True.
    u : TYPE, optional  （不推荐使用!）
        因为不同的栅格数据或是来源或是处理不同，如配准手动配准的两个栅格大概率无法对齐，又或是处理不够规范，栅格是不对齐的。
        栅格对齐没搞明白，不敢乱动，这个clip处理后不对齐的两个栅格范围的偏差不会超过一个栅格，一般没有什么关系
        因栅格不对齐，所以无法确保范围完全一致，(误差在1个栅格大小内)
        
        这个填True者是使得范围显示的统一，但矩阵没变而且在arcmap中可以看到偏移，就是说这个参数没啥用
        . The default is False.

    Returns
    -------
    arr : TYPE
        栅格矩阵
    profile : TYPE
        栅格属性

    '''
    
    # 获取栅格
    src = rasterio.open(raster_in) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
    dst = rasterio.open(dst_in) if issubclass(type(dst_in), (str,pathlib.PurePath)) else dst_in

    
    # 请保证数据空间参考一致
    if dst.crs != src.crs:
        TypeError("源数据与目标数据空间参考不一致")
    
    if win_shape:
        if not(bool(out_path) | bool(get_ds)):
            ValueError("使用win_shape、win_size分窗口参数时只能返回栅格类及输出栅格,请按需填入out_path、get_ds参数")
    

    
    # 获取目标位置在源数据中的对应窗口,及新的行列数
    nodata = src.nodata
    profile = src.profile
    
    left0, bottom0, right0, top0 = src.bounds
    left, bottom, right, top = bounds or dst.bounds
    
    
    inter = (max(left,left0),
             max(bottom,bottom0),
             min(right,right0),
             min(top,top0))
    
    if (inter[2] <= inter[0]) | (inter[3] <= inter[1]):
        # print('输入范围与栅格不重叠')
        if inner:
            Exception('\nclip: 输入范围与栅格不重叠')
        else:
            warnings.warn('\nclip: 输入范围与栅格不重叠')
    
    
    if inner:
        left, bottom, right, top = inner
    else:
        pass
    
    
    y_off,x_off = src.index(left, top)
    y_end, x_end = src.index(right, bottom)
    
    
    
    if unify and ap.get_RasterAttr(src,'cell_size') == ap.get_RasterAttr(dst,'cell_size'):
        width, height = dst.shape[::-1]
    else:
        width, height = x_end-x_off, y_end-y_off
    win = Window(x_off, y_off,width, height)
    
    # 获取transform(仿射变换)
    if u:
        transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    else:
        transform = src.window_transform(win)
    
    # 更新profile
    profile.update({'transform':transform, 'width':width, 'height':height})
    
    # 返回处理
    if win_shape:
        # 分窗口写出数据
        dst_win = create_raster(**profile)
        src_windows,_ = window(dst_win,shape=win_shape,initial_offset=(x_off, y_off))  # 数据在源栅格的窗口位置
        dst_windows,_ = window(dst_win,shape=win_shape)  # 数据在输出栅格的窗口位置，与源栅格对应
        
        # 输出
        if out_path:
            with rasterio.open(out_path, 'w', **profile) as ds:
                if Tqbm:
                    pbar = tqdm(total=win_shape[0]*win_shape[1], desc='clip')
                for i in range(len(src_windows)):
                    src_win = src_windows[i]
                    dst_win = dst_windows[i]
                    arrx = src.read(window=src_win,boundless=True,fill_value=nodata)
                    ds.write(arrx,window=dst_win)
                    pbar.update(1)
                if Tqbm:
                    pbar.close()
        elif get_ds:
            ds = create_raster(**profile)
            if Tqbm:
                pbar = tqdm(total=win_shape[0]*win_shape[1], desc='clip')
            for i in range(len(src_windows)):
                src_win = src_windows[i]
                dst_win = dst_windows[i]
                arrx = src.read(window=src_win,boundless=True,fill_value=nodata)
                ds.write(arrx,window=dst_win)
                pbar.update(1)
            if Tqbm:
                pbar.close()
            return ds
        else:
            Exception("有问题")
        # 可两皆满足但感觉不太会有这种需求，多占内存与运行，并不启用
        # dss = []
        # if get_ds:
        #     ds_on = create_raster(**profile)
        #     dss.append(ds_on)
        # if out_path:
        #     ds_out = rasterio.open(out_path, 'w', **profile)
        #     dss.append(ds_out)
        
        # for i in range(len(src_windows)):
        #     src_win = src_windows[i]
        #     dst_win = dst_windows[i]
        #     arrx = src.read(window=src_win,boundless=True,fill_value=nodata)
        #     [ds.write(arrx,window=dst_win) for ds in dss]
        
    else:
        arr = src.read(window=win,boundless=True,fill_value=nodata)
        np.unique(arr)
        if out_path:
            with rasterio.open(out_path,'w',**profile) as ds:
                ds.write(arr)
        elif get_ds:
            ds = create_raster(**profile)
            ds.write(arr)
            return ds
        else:
            return arr, profile


def resampling(raster_in, out_path, re_shape=None, re_size=None, win_shape=None, how='mode', Tqbm=True):
    '''
    分窗口重采样
    
    Parameters
    ----------
    raster_in : TYPE
        源栅格数据或地址
    out_path : TYPE
        输出路径
    re_shape : TYPE, optional
        重采样目标形状
    win_shape : TYPE, optional
        分的窗口形状
    how:(str or int) , optional.
    重采样方式，The default is mode.
    (部分)
    mode:众数，6;
    nearest:临近值，0;
    bilinear:双线性，1;
    cubic_spline:三次卷积，3。
    ...其余见rasterio.enums.Resampling

    Returns
    -------
    None.

    '''
    
    
    src = rasterio.open(raster_in) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
    
    
    # 创建输出栅格
    profile = src.profile
    left, bottom, right, top = src.bounds
    if re_size:
        if (type(re_size) == int) | (type(re_size) == float):
            xsize = re_size
            ysize = re_size
        else:
            xsize, ysize = re_size
        transform = rasterio.transform.from_origin(left,top, xsize, ysize)
        height, width = np.ceil((top - bottom) / ysize), np.ceil((right - left) / xsize)
    else:
        height, width = re_shape
        transform = rasterio.transform.from_bounds(*src.bounds, width, height)
    profile.update({'transform':transform, 'width':width, 'height':height})
    
    dst = rasterio.open(out_path, 'w', **profile)
    
    # 获取源数据与输出数据对应窗口
    win_shape = win_shape or (1,1)
    windows_src, _ = window(src,shape=win_shape)
    windows_dst, _ = window(dst,shape=win_shape)
    if Tqbm:
        pbar = tqdm(total=win_shape[0]*win_shape[1], desc='resampling')
    
    # 分窗口重采样
    for i in range(len(windows_src)):
        
        win_src = windows_src[i]
        win_dst = windows_dst[i]
        
        out_shape = (win_dst.height,win_dst.width)  # 单窗口目标形状
        how = how if isinstance(how, int) else getattr(Resampling, how)  # 重采样方法
        arr = src.read(window=win_src, out_shape=out_shape,resampling=how)  # 重采样
        
        dst.write(arr,window=win_dst)  # 写出
        if Tqbm:
            pbar.update(1)
    if Tqbm:
        pbar.close()
    














if __name__ == "__main__":

    raster_in = r'F:/PyCharm/pythonProject1/代码/mycode/测试文件/源数据/1990-5km-tiff.tif'
    
    no = r'F:/PyCharm/pythonProject1/代码/mycode/测试文件/源数据/eva_2.tif'
    r = r'F:/PyCharm/pythonProject1/代码/mycode/测试文件/源数据/eva.tif'
    dst_in = r
    

    # dst = ap.reproject(dst_in,raster_in,out_path=r'F:/PyCharm/pythonProject1/代码/mycode/测试文件/源数据/eva.tif')
    dst = rasterio.open(dst_in)
    
    # dst.window_transform(Window(-100,-100,10,10))
    # dst.transform
    # x = dst.read(window=Window(-100,-100,10,10),fill_value=np.nan,boundless=True)
    # dst.bounds
    # 输出
    out_path1 = r'F:\PyCharm\pythonProject1\代码\mycode\测试文件\clip_1.tif'
    out_path2 = r'F:\PyCharm\pythonProject1\代码\mycode\测试文件\clip_2.tif'
    
    clip(raster_in,dst,out_path=out_path1,u=0,win_shape=(3,3))
    
    
    ap.clip(raster_in,dst,out_path=out_path2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # out_path1 = r'F:\PyCharm\pythonProject1\代码\mycode\测试文件\重采样_1.tif'
    # out_path2 = r'F:\PyCharm\pythonProject1\代码\mycode\测试文件\重采样_2.tif'
    
    


    # out_shape_sum = (1368*3,1728*3)  # (height,width)
    
    
    # resampling(raster_in,out_path1,re_shape=out_shape_sum,how=3,win_shape=(3,3))
        
    # # dst.close()
    
    # ap.resampling(raster_in, re_shape=out_shape_sum,out_path=out_path2,how=3)
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    











































