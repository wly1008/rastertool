# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:11:21 2024

@author: wly
"""


import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.coords import disjoint_bounds
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds
from rasterio.transform import array_bounds
from rasterio.warp import reproject

import cv2
import math
import warnings
import numpy as np
import pathlib, os
from contextlib import ExitStack
from shapely.geometry import box


from rastertool.warnings import SetNodataWarning, SetRasterAttrWarning


def eq_crs(crs1, crs2):
    return CRS.from_user_input(crs1) == CRS.from_user_input(crs2)

def bounds_intersection(bounds1, bounds2):
    box1 = box(*bounds1)  # 创建第一个矩形
    box2 = box(*bounds2)  # 创建第二个矩形
    
    intersection = box1.intersection(box2)
    return intersection.bounds
    

def create_raster(**kwargs):
    memfile = rasterio.MemoryFile()
    return memfile.open(**kwargs)

def align(array, pad_width, mode='constant',constant_values=None, **kwargs):
    '''在np.pad基础上加入负值处理,
       pad_width中负值代表剔除,正值代表填充
       参数说明见np.pad
    '''
    pad_width = np.asarray(pad_width)
    
    # 替换负值运行np.pad
    _pad_width = pad_width.copy()
    _pad_width[_pad_width<0] = 0
    
    ret0 = np.pad(array, _pad_width, mode=mode, constant_values=constant_values, **kwargs)

    # 将负值理解为剔除，计算切片
    slices = []
    
    r = pad_width[:,0]
    l = pad_width[:,1]
    for dim in range(ret0.ndim):
        if r[dim] < 0 :
            start = -r[dim]
        else:
            start = 0
        if l[dim] >= 0:
            stop = None
        else:
            stop = l[dim]
        
        slices.append(slice(start, stop))
        
    return ret0[tuple(slices)]

def tr_winattr(off, length,*funcs):
    '''调用取整方法，并消除off改变对length+off的影响'''
    off_tr = funcs[0](off)
    length_tr = off + length - off_tr
    # if axis == 'col':
    #     length_tr = off + length - off_tr
    # elif axis == 'row':
    #     length_tr = off_tr - (off - length)
    length_tr = funcs[-1](length_tr)
    return int(off_tr), int(length_tr)


def _round(x):
    '''避免内置`round`0.5向偶数方向取值问题'''
    return math.floor(x + 0.5)


# 窗口取整函数
def round_window(win):
    '''四舍五入'''
    col_off, row_off, width, height = win.col_off, win.row_off, win.width, win.height
    
    col_off, width = tr_winattr(col_off, width, _round)
    row_off, height = tr_winattr(row_off, height, _round)
    
    
    return Window(col_off, row_off, width, height)


def rasterio_window(win):
    '''rio.clip标准'''
    
    out_window = win.round_lengths()  # int(math.floor(x + 0.5)) 四舍五入
    out_window = out_window.round_offsets()  # int(math.floor(x + 0.001)) 近乎向下取整，在0.001范围向上取
    
    return out_window



def touch_window(win):
    '''保留所有接触像元'''
    col_off, row_off, width, height = win.col_off, win.row_off, win.width, win.height
    
    col_off, width = tr_winattr(col_off, width, *(math.floor, math.ceil))
    row_off, height = tr_winattr(row_off, height, *(math.floor, math.ceil))
    
    
    return Window(col_off, row_off, width, height)

def touch_arr(arr_crop,src_transform, dst_transform, nodate=0):
    
    arr_right = np.pad(arr_crop,[[0, 0], [0, 1], [1, 0]],
                       mode='constant', constant_values=0)
    arr_down = np.pad(arr_crop,[[0, 0], [1, 0], [0, 1]],
                       mode='constant', constant_values=0)
    
    arr_crop = arr_crop | arr_down | arr_right
    arr_crop[:,-1,-1] = arr_crop[:,-2,-2]
    return arr_crop

def round_bounds(left, bottom, right, top, src_width, src_height):
    bounds = (left, bottom, right, top)
    return tuple([_round(i) for i in bounds])


def riomode_bounds(left, bottom, right, top, src_width, src_height):
    
    width = left + right + src_width  # New_width
    height = top + bottom + src_height  # New_height
    _left = math.floor(left + 0.5)
    _top = math.floor(top + 0.5)
    _right = math.floor(width + 0.001) - _left - src_width
    _bottom = math.floor(height + 0.001) - _top - src_height
    
    return (_left, _bottom, _right, _top)

def touch_bounds(left, bottom, right, top, src_width, src_height):
    bounds = (left, bottom, right, top)
    return tuple([math.ceil(i) for i in bounds])


def _clip_array(source, bounds, src_bounds=None, src_transform=None,
                mode='round', filled=False):
    '''
    栅格矩阵裁剪

    Parameters
    ----------
    source : np.ma.array
        
    bounds : TYPE
        DESCRIPTION.
    src_bounds : TYPE, optional
        DESCRIPTION. The default is None.
    src_transform : TYPE, optional
        DESCRIPTION. The default is None.
    mode : TYPE, optional
        DESCRIPTION. The default is 'round'.
    filled : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    out_arr : TYPE
        DESCRIPTION.
    out_transform : TYPE
        DESCRIPTION.

    '''
    if source.ndim == 2:
        src_count = 1
        src_height, src_width = source.shape
    else:
        src_cound, src_height, src_width = source.shape
    
    if src_bounds and src_transform:
        raise ValueError('`src_bounds`,`src_transform`, 同时存在, 请选择其中之一使用')
    
    if src_transform is None:
        if src_transform is None:
            raise ValueError('请输入`src_bounds`或`src_transform`')
        src_transform = from_bounds(*src_bounds, src_width, src_height)
    else:
        src_bounds = array_bounds(src_height, src_width, src_transform)
    
    west, south, east, north = src_bounds

    srcx = src_transform[0]
    srcy = src_transform[4]
    
    sl, sb, sr, st = src_bounds
    dl, db, dr, dt = bounds
    
    # 检查方向
    if not ((srcx >= 0) == (dr >= dl)):
        dr, dl = dl, dr
    
    if not ((srcy <= 0) == (db <= dt)):
        db, dt = dt, db
    
    # 检查是否重叠

    if disjoint_bounds((dl, db, dr, dt), src_bounds):
        raise ValueError('must overlap the extent of '
                                 'the input raster')
    
    # TODO
    # col
    left = (sl - dl) / srcx
    right = (dr - sr) / srcx
    
    # row
    top = (st - dt) / srcy
    bottom = (db - sb) / srcy

    if mode == 'round':
        left, bottom, right, top = round_bounds(left, bottom, right, top, src_width, src_height)
    
    elif mode == 'rio':
        left, bottom, right, top = riomode_bounds(left, bottom, right, top, src_width, src_height)
        
    elif mode == 'touch':
        left, bottom, right, top = touch_bounds(left, bottom, right, top, src_width, src_height)
    
    elif callable(mode):
        left, bottom, right, top = mode(left, bottom, right, top, src_width, src_height)
    else:
        raise ValueError('mode 可选参数为 round,rio,touch,或者输入自定义矩阵范围取整函数')

    top+ bottom + src_height
    col = [left, right]
    row = [top, bottom]
    
    out_arr_data = align(source.data, [[0, 0], row, col], mode='constant', constant_values=source.fill_value)
    if np.isscalar(source.mask):
        out_arr_mask = np.full_like(out_arr_data, source.mask)
    else:
        out_arr_mask = align(source.mask, [[0, 0], row, col], mode='constant', constant_values=True)
    
    out_arr = np.ma.masked_array(out_arr_data, mask=out_arr_mask)
    
    out_transform = from_origin(west - left*srcx, north - top*srcy, srcx, -srcy)
    
    return out_arr, out_transform


def clip(raster_in, dst_in=None, out_path=None,
         get_ds=False,
         bounds=None, 
         mode='round',
         nodata=None,
         dtype=None,
         crs='EPSG:4326',
         filled=True,
         with_complement=True,
         crop=False,
         arr_crop=None,
         crop_use_index=None,
         ushape=False,
         shape=None,
         delete=False,
         stacklevel=2,
         **creation_options,
         ):
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
    get_ds : bool, optional
        是否获取临时栅格.当out_path为None时有效. Default: True.
    bounds : list、tuple, optional
        目标范围，(左，下，右，上). The default is None.
    mode : str or function, optional
        裁剪模式，默认为round，可自定义，参考上方round_window等
        round: 
            四舍五入
        rio: rio.clip标准，
            lengths:int(math.floor(x + 0.5)) 四舍五入
            offsets:int(math.floor(x + 0.001)) 近乎向下取整，在0.001范围向上取
        touch: 
            保留所有接触像元
    nodata : 数字类, optional
        输出栅格无效值,为None时与源栅格一致, 若源栅格没有无效值, nodata取0. The default is None.
    dtype : str or numpy dtype opt
        输出栅格值类型,为None时与源栅格一致, 默认为None
    crs : CRS or dict, optional
        输入范围的空间参考. The default is 'EPSG:4326'.
    with_complement : TYPE, optional
        是否补足区并集. The default is True.
    crop : bool, optional
        是否对目标有效值进行提取
        The default is True.
    arr_crop : array, optional
        有效值掩膜数组，如已输入dst_in请忽略
        The default is None.
    crop_use_index : int or slice
        arr_crop使用的第一个纬度切片

    delete : TYPE, optional
        是否删除输入栅格raster_in（清除中间变量）,
        当raster_in为地址时正常执行, 而输入栅格类变量时不会, 除非使用'!True'.
        The default is False.
    stacklevel : int opt
        警告级别，默认为3，详见warnings.warn
    **creation_options : TYPE
        输出栅格其他profile更新选项



    Returns
    -------
    if out_path:生成栅格文件，返回文件地址
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重投影后的栅格矩阵（array）和 profile
    
    '''
    
    
    if dst_in is None:
        if bounds is None:
            raise ValueError('please input dst_in or bounds')
        
        if crop and arr_crop is None:
            raise ValueError('crop is True, please input dst_in or arr_crop')
        if ushape and shape is None:
            raise ValueError('ushape is True, please input dst_in or shape')
    
    
    if crop:

        if crop_use_index is None:
            crop_use_slice = slice(None)
        elif isinstance(crop_use_index, int):
            crop_use_slice = slice(crop_use_index, crop_use_index+1)
        elif isinstance(crop_use_index, slice):
            crop_use_slice = crop_use_index
        else:
            raise TypeError(f'--crop_use_index is {type(crop_use_index)},but it must be an int or a slice.')
    
    

    
    with ExitStack() as stack:
        
        # 
        src = stack.enter_context(rasterio.open(raster_in)) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
        
        
        if not src.transform.is_rectilinear:
            raise ValueError(
                "Non-rectilinear rasters (i.e. with rotation or shear) cannot be clipped"
            )
    

        if dst_in:
            
            template_ds = stack.enter_context(rasterio.open(dst_in)) if issubclass(type(dst_in), (str,pathlib.PurePath)) else dst_in
            bounds = template_ds.bounds
            if template_ds.crs != src.crs:
                bounds = transform_bounds(template_ds.crs, src.crs,
                                          *bounds)
            crs = template_ds.crs
        elif bounds:
            bounds = transform_bounds(crs, src.crs, *bounds)
        else:
            raise ValueError('--bounds or --dst_in required')
        
        
            
        # 检查是否重叠
        if disjoint_bounds(bounds, src.bounds):
            raise ValueError('must overlap the extent of '
                                     'the input raster')
        
        bounds_window = src.window(*bounds)

    
        if not with_complement:
            bounds_window = bounds_window.intersection(
                Window(0, 0, src.width, src.height)
            )
        
        
        # Align window
        
        if mode == 'rio':
            
            out_window = rasterio_window(bounds_window)
        elif mode == 'round':
            
            out_window = round_window(bounds_window)
        elif mode == 'touch':
            
            out_window = touch_window(bounds_window)
        
        elif callable(mode):
            out_window = mode(bounds_window)
        else:
            raise ValueError('mode 可选参数为 round,rio,touch,或者输入自定义窗口取整函数')
        
        
        
        if ushape:
            shape = shape or template_ds.shape
            
            if len(shape) == 3:
                shape = shape[1:]
            elif len(shape) == 2:
                pass
            else:
                raise ValueError('Only 2D or 3D shapes are accepted, but %dD is received'%len(shape))
            height, width = shape
            out_window = Window(out_window.col_off, out_window.row_off, width, height)
        
        

        
        height = int(out_window.height)
        width = int(out_window.width)
    
        out_kwargs = src.profile.copy()
        transform = src.window_transform(out_window)
        # src_nodata = src.nodata
        bandNames = src.descriptions
        
        
        out_kwargs.update({
            'height': height,
            'width': width,
            'transform': transform})
        
        out_kwargs.update(descriptions=bandNames,
                          **creation_options)
        
        if "blockxsize" in out_kwargs and int(out_kwargs["blockxsize"]) > width:
            del out_kwargs["blockxsize"]
            warnings.warn(
                "Blockxsize removed from creation options to accomodate small out_path width",
                SetRasterAttrWarning,
                stacklevel=stacklevel,
            )
        if "blockysize" in out_kwargs and int(out_kwargs["blockysize"]) > height:
            del out_kwargs["blockysize"]
            warnings.warn(
                "Blockysize removed from creation options to accomodate small out_path height",
                SetRasterAttrWarning,
                stacklevel=stacklevel,
                )
        # TODO
        if crop:
            if not eq_crs(crs, src.crs):
                raise ValueError('crop 仅支持在相同空间参考中数组使用')
        
        arr = src.read(
                        window=out_window,
                        out_shape=(src.count, height, width),
                        boundless=True,
                        masked=True,
                        )

        if crop:
            if not eq_crs(crs, src.crs):
                raise ValueError('crop 仅支持在相同空间参考中数组使用')
            win_right = round_window(bounds_window)
            if dst_in:
                
                arr_crop = template_ds.read_masks(out_shape=(template_ds.count, win_right.height, win_right.width))
                
            
            elif arr_crop is None:
                raise ValueError('--arr_crop or --dst_in required')
            
            if arr_crop.ndim == 2:
                arr_crop = np.asarray([arr_crop])
            arr_crop = arr_crop[crop_use_slice]
            if not ((arr_crop.shape[0] == 1) or (arr_crop.shape[0] == arr.shape[0])):
                raise ValueError('arr_crop 与 arr波段数不匹配且arr_crop不为单波段, 无法广播, 可尝试使用 crop_use_index 参数')
            
            # 计算调整 arr_crop, 使与 arr对齐, 此处假设 arr_crop位置与 round_window(bounds_window)一致
            if arr_crop.shape[1:] != arr.shape[1:]:
                '''添加一些检查，自定义的mode可能改变窗口初始位置'''
                win_right = round_window(bounds_window)
                
                # col
                left = win_right.col_off - out_window.col_off
                right = (out_window.col_off + out_window.width) - (win_right.col_off + win_right.width)
                col = [left, right]

                
                # row
                top = win_right.row_off - out_window.row_off
                bottom = (out_window.row_off + out_window.height) - (win_right.row_off + win_right.height)
                row = [top, bottom]

                arr_crop = align(arr_crop, [[0, 0], row, col],mode='constant',constant_values=0)
                
            arr.mask = np.where(arr_crop == 0, True, arr.mask)
            # arr = np.where(arr_crop == 0, src_nodata, arr)

    
    # 删除中间栅格
    if delete and issubclass(type(raster_in), (str,pathlib.PurePath)):
        
        os.remove(raster_in)
    elif delete == '!True':
        file = raster_in.files[0]
        raster_in.close()
        os.remove(file)
    
    
    # 更新无效值
    if nodata is None:
        nodata = out_kwargs['nodata']
    
    if nodata is None:
        nodata = 0
        warnings.warn(
            "源数据无nodata, 自动设定为0, 注意可能产生的影响。 如非所需请为参数nodata赋值",
            SetNodataWarning,
            stacklevel=stacklevel
                        )
    out_kwargs['nodata'] = nodata
    
    # 更新`arr`的类型
    if dtype is not None:
        arr = arr.astype(dtype)
        out_kwargs['dtype'] = dtype
    
    # 检查无效值能否储存在`arr`中
    dtype = arr.dtype
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
        with rasterio.open(out_path, "w", **out_kwargs) as out:
            out.write(arr)
            out.descriptions = bandNames

        return out_path
    elif get_ds:
        out = create_raster(**out_kwargs)
        out.write(arr)
        out.descriptions = bandNames

        return out
    else:
        return (arr, out_kwargs)












# def reproject(
#     source,
#     destination=None,
#     src_transform=None,
#     gcps=None,
#     rpcs=None,
#     src_crs=None,
#     src_nodata=None,
#     dst_transform=None,
#     dst_crs=None,
#     dst_nodata=None,
#     dst_resolution=None,
#     src_alpha=0,
#     dst_alpha=0,
#     masked=False,
#     resampling=Resampling.nearest,
#     num_threads=1,
#     init_dest_nodata=True,
#     warp_mem_limit=0,
#     src_geoloc_array=None,
#     **kwargs
# ):...



def check_nodataIsNone(nodata, name,stacklevel=2):
    
    if nodata is None:
        nodata = 0
        warnings.warn(
                    "未输入 %s , 自动设定为0, 注意可能产生的影响。 如非所需请为参数 %s 赋值"%(name,name),
                    SetNodataWarning,
                    stacklevel=stacklevel
                    )
    return nodata


def clip_array(
    source,
    destination=None,
    src_bounds=None,
    src_transform=None,
    src_crs='EPSG:4326',
    src_nodata=None,
    dst_bounds=None,
    dst_transform=None,
    dst_height=None,
    dst_width=None,
    dst_crs='EPSG:4326',
    dst_nodata=None,
    nodata=None,
    mode='round',
    
    crop=False,
    crop_use_index=None,
    
    with_complement=True,
    masked=False,
    
    stacklevel=2,
    ):
    
    
    if dst_crs is None:
        dst_crs = src_crs
    
    
    # 检查nodata
    src_nodata = check_nodataIsNone(src_nodata,'src_nodata',stacklevel=stacklevel+1)
    if crop:
        dst_nodata = check_nodataIsNone(dst_nodata,'dst_nodata',stacklevel=stacklevel+1)
    
    if nodata is None:
        nodata = src_nodata
    
    
    if not hasattr(source, 'mask'):
        source = np.ma.masked_equal(source, src_nodata)
    else:
        source.fill_value = src_nodata
    
    if source.ndim == 2:
        source = np.ma.asarray([source])
    src_count, src_height, src_width = source.shape
    
        

    
    
    if crop:
        # 参数条件检查
        if destination is None:
            raise ValueError('crop is True, please input destination')
        
        if not (dst_height is None and dst_width is None):
            warnings.warn('`destination` 存在时, dst_height, dst_width无效',
                          UserWarning,
                          stacklevel=stacklevel)
        
        # destination 设置
        if not hasattr(destination, 'mask'):
            destination = np.ma.masked_equal(destination, dst_nodata)
        if destination.ndim == 2:
            destination = np.ma.asarray([destination])
        dst_count, dst_height, dst_width = destination.shape
        
        if not eq_crs(src_crs, dst_crs):
            raise ValueError('crop 仅支持在相同空间参考中数组使用')
            
        # destinatio设置
        if not hasattr(destination, 'mask'):
            destination = np.ma.masked_equal(destination, dst_nodata)
        if destination.ndim == 2:
            destination = np.ma.asarray([destination])
        
        # 选取波段
        if crop_use_index is None:
            crop_use_slice = slice(None)
        elif isinstance(crop_use_index, int):
            crop_use_slice = slice(crop_use_index, crop_use_index+1)
        elif isinstance(crop_use_index, slice):
            crop_use_slice = crop_use_index
        else:
            raise TypeError(f'--crop_use_index is {type(crop_use_index)},but it must be an int or a slice.')
        destination = destination[crop_use_slice]
        dst_count, dst_height, dst_width = destination.shape
        
        if dst_count != 1 and dst_count != src_count:
            raise ValueError('source 与 destination 波段数不匹配且 destination 不为单波段, 无法广播, 可尝试使用 crop_use_index 参数')
        
        crop_mask = np.full_like(destination, destination.mask) if np.isscalar(destination.mask) else destination.mask
        if dst_nodata is np.nan:
            destination.mask = np.where(np.isnan(destination), True, crop_mask)
    
    
    if src_bounds and src_transform:
        raise ValueError('src_bounds and src_transform are mutually'
                         "exclusive parameters and may not be used together."
                         )
    
    if src_bounds is None:
        if src_transform is None:
            raise ValueError("`src_bounds`和`src_transform`, 需且仅需输入其中一个")
        
        src_bounds = array_bounds(src_height, src_width, src_transform)
    else:
        # sl, sb, sr, st = src_bounds
        src_transform = from_bounds(*src_bounds, src_width, src_height)

    
    if dst_bounds and dst_transform:
        raise ValueError('dst_bounds and dst_transform are mutually'
                         "exclusive parameters and may not be used together."
                         )
        
    if dst_bounds is None:
        if dst_transform is None:
            raise ValueError("`dst_bounds`和`dst_transform`, 需且仅需输入其中一个")
        
        if destination is None:
            
            # 1. dst_transform, dst_height, dst_width
            
            if None in {dst_height, dst_width}:
                
                err_msg = '未接受到`dst_bounds`、`destination`。 且`dst_height`(%s), `dst_width`(%s)中存在None, 无法计算目标范围。要求详见函数说明'
                raise ValueError(err_msg % (dst_height, dst_width))
            
        else:
            # 2. dst_transform and destination
            if not (dst_height is None and dst_width is None):
                if not crop:
                    warnings.warn('`destination` 存在时, dst_height, dst_width无效',
                                  UserWarning,
                                  stacklevel=stacklevel)
            if destination.ndim == 3:
                dst_count, dst_height, dst_width = destination.shape
            else:
                dst_count = 1
                dst_height, dst_width = destination.shape
        
        
        dst_bounds = array_bounds(dst_height, dst_width, dst_transform)
    
    else:
        if crop:
            dst_transform = from_bounds(*dst_bounds, dst_height, dst_width)
    
    
    bounds = transform_bounds(dst_crs, src_crs,
                              *dst_bounds)
    

    bounds = bounds_intersection(src_bounds, bounds) if not with_complement else bounds
    
    out_arr, out_transform = _clip_array(source, bounds,src_transform=src_transform, mode=mode)
    
    if crop:
        
        if not with_complement:
            dst_arr, _ = _clip_array(destination, bounds,src_transform=dst_transform, mode=mode)
            crop_mask = dst_arr.mask
            
        if crop_mask.shape[1:] != out_arr.shape[1:]:
            crop_mask = crop_mask.transpose((1, 2, 0))
            crop_mask = cv2.resize(crop_mask.astype('uint8'), out_arr.shape[-1:0:-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            if crop_mask.ndim == 3:
                crop_mask = crop_mask.transpose((2, 0, 1))
                
        out_arr.mask = np.where(crop_mask == True, True, out_arr.mask)
        
    
    out_arr_data = out_arr.filled(nodata)
    
    if masked:
        out_arr = np.ma.array(out_arr_data, mask=out_arr.mask)
        
    else:
        out_arr = out_arr_data
    
    return out_arr, out_transform





def clip_array_crop(
    source,
    destination,
    src_bounds,
    dst_bounds,
    src_crs='EPSG:4326',
    src_nodata=None,

    dst_crs='EPSG:4326',
    dst_nodata=None,
    nodata=None,
    mode='round',
    
    crop_use_index=None,
    
    with_complement=True,
    masked=False,
    
    stacklevel=2,
    ):
    
    # 检查 crs
    if src_crs is None:
        raise ValueError('Please input `src_crs`')
    
    if dst_crs is None:
        dst_crs = src_crs
    
    
    # 检查 nodata
    src_nodata = check_nodataIsNone(src_nodata,'src_nodata',stacklevel=stacklevel+1)
    
    dst_nodata = check_nodataIsNone(dst_nodata,'dst_nodata',stacklevel=stacklevel+1)
    
    if nodata is None:
        nodata = src_nodata
    
    
    # 设置 source
    if not hasattr(source, 'mask'):
        source = np.ma.masked_equal(source, src_nodata)
    else:
        source.fill_value = src_nodata
    if source.ndim == 2:
        source = np.ma.asarray([source])
    src_count, src_height, src_width = source.shape
    
    # 设置 destination 
    if not hasattr(destination, 'mask'):
        destination = np.ma.masked_equal(destination, dst_nodata)
    else:
        destination.fill_value = dst_nodata
    if destination.ndim == 2:
        destination = np.ma.asarray([destination])
    
    # 选取波段
    if crop_use_index is None:
        crop_use_slice = slice(None)
    elif isinstance(crop_use_index, int):
        crop_use_slice = slice(crop_use_index, crop_use_index+1)
    elif isinstance(crop_use_index, slice):
        crop_use_slice = crop_use_index
    else:
        raise TypeError(f'--crop_use_index is {type(crop_use_index)},but it must be an int or a slice.')
    destination = destination[crop_use_slice]
    dst_count, dst_height, dst_width = destination.shape
    
    if dst_count != 1 and dst_count != src_count:
        raise ValueError('source 与 destination 波段数不匹配且 destination 不为单波段, 无法广播, 可尝试使用 crop_use_index 参数')
    
    
    
    
    # transform
    src_transform = from_bounds(*src_bounds, src_width, src_height)
    dst_transform = from_bounds(*dst_bounds, dst_width, dst_height)
    
    # ras
    # src_transform = from_bounds(0, 0, 10, 10, 10, 10)
    src_ras = np.abs(src_transform[:5:4]).tolist()
    dst_ras = np.abs(dst_transform[:5:4]).tolist()
    if not eq_crs(src_crs, dst_crs):
        destination, dst_transform = reproject(
                                               source=destination.filled(dst_nodata),
                                               src_nodata=dst_nodata,
                                               src_transform=dst_transform,
                                               src_crs=dst_crs,
                                               
                                               dst_crs=src_crs,
                                               dst_nodata=dst_nodata,
                                               dst_resolution=src_ras,
                                               resampling=Resampling.nearest,
                                               
                                               masked=True,
                                               num_threads=4,
                                               
                                               )
        dst_count, dst_height, dst_width = destination.shape
        dst_bounds = array_bounds(dst_height, dst_width, dst_transform)
    
    
    bounds = bounds_intersection(src_bounds, dst_bounds) if not with_complement else dst_bounds
    
    out_arr, out_transform = _clip_array(source, bounds,src_transform=src_transform, mode=mode)
    
    if not with_complement:
        destination, dst_transform = _clip_array(destination, bounds,src_transform=dst_transform, mode=mode)
    crop_mask = destination.mask
    
    if crop_mask.shape[1:] != out_arr.shape[1:]:
        crop_mask = crop_mask.transpose((1, 2, 0))
        crop_mask = cv2.resize(crop_mask.astype('uint8'), out_arr.shape[-1:0:-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        if crop_mask.ndim == 3:
            crop_mask = crop_mask.transpose((2, 0, 1))
            
    out_arr.mask = np.where(crop_mask == True, True, out_arr.mask)
    
    
    out_arr_data = out_arr.filled(nodata)
    
    if masked:
        out_arr = np.ma.array(out_arr_data, mask=out_arr.mask)
        
    else:
        out_arr = out_arr_data
    
    return out_arr, out_transform
    

def clip_array(
    source,
    src_bounds,
    dst_bounds,

    src_crs='EPSG:4326',
    src_nodata=None,

    dst_crs='EPSG:4326',
    nodata=None,
    mode='round',
    
    with_complement=True,
    masked=False,
    
    stacklevel=2,
    ):
    
    # 检查 crs
    if src_crs is None:
        raise ValueError('Please input `src_crs`')
    
    if dst_crs is None:
        dst_crs = src_crs
    
    if not hasattr(source, 'mask'):
        source = np.ma.masked_equal(source, src_nodata)
    else:
        source.fill_value = src_nodata
    
    if source.ndim == 2:
        source = np.ma.asarray([source])
    src_count, src_height, src_width = source.shape
    
    
    bounds = transform_bounds(dst_crs, src_crs,
                              *dst_bounds)
    
    intersection = bounds_intersection(src_bounds, bounds)
    
    if np.isnan(intersection).sum() == 4:
        raise ValueError('must overlap the extent of '
                                 'the input raster')
        
    bounds = intersection if not with_complement else bounds
    
    out_arr, out_transform = _clip_array(source, bounds,src_bounds=src_bounds, mode=mode)
    
    
    
    out_arr_data = out_arr.filled(nodata)
    
    if masked:
        out_arr = np.ma.array(out_arr_data, mask=out_arr.mask)
        
    else:
        out_arr = out_arr_data
    
    return out_arr, out_transform
    
    
    
    

def clip_rasterArray(
                     source,
                     destination=None,
                     src_bounds=None,
                     src_transform=None,
                     src_crs='EPSG:4326',
                     src_nodata=None,
                     dst_bounds=None,
                     dst_transform=None,
                     dst_height=None,
                     dst_width=None,
                     dst_crs='EPSG:4326',
                     dst_nodata=None,
                     nodata=None,
                     mode='round',
                     
                     crop=False,
                     crop_use_index=None,
                     
                     with_complement=True,
                     masked=False,
                     
                     stacklevel=2,
                     ):
    # 检查 crs
    if src_crs is None:
        raise ValueError('Please input `src_crs`')
    
    if dst_crs is None:
        dst_crs = src_crs
    
    # 设置source
    if not hasattr(source, 'mask'):
        source = np.ma.masked_equal(source, src_nodata)
    else:
        source.fill_value = src_nodata
    
    if source.ndim == 2:
        source = np.ma.asarray([source])
    src_count, src_height, src_width = source.shape
    
    
    if src_bounds and src_transform:
        raise ValueError('src_bounds and src_transform are mutually'
                         "exclusive parameters and may not be used together."
                         )
    
    if src_bounds is None:
        if src_transform is None:
            raise ValueError("`src_bounds`和`src_transform`, 需且仅需输入其中一个")

        src_bounds = array_bounds(src_height, src_width, src_transform)

    
    if dst_bounds and dst_transform:
        raise ValueError('dst_bounds and dst_transform are mutually'
                         "exclusive parameters and may not be used together."
                         )
        
    if dst_bounds is None:
        if dst_transform is None:
            raise ValueError("`dst_bounds`和`dst_transform`, 需且仅需输入其中一个")
        
        if destination is None:
            
            # 1. dst_transform, dst_height, dst_width
            
            if None in {dst_height, dst_width}:
                
                err_msg = '未接受到`dst_bounds`、`destination`。 且`dst_height`(%s), `dst_width`(%s)中存在None, 无法计算目标范围。要求详见函数说明'
                raise ValueError(err_msg % (dst_height, dst_width))
            
        else:
            # 2. dst_transform and destination
            if not (dst_height is None and dst_width is None):
                warnings.warn('`destination` 存在时, dst_height, dst_width无效',
                              UserWarning,
                              stacklevel=stacklevel)
            if destination.ndim == 3:
                dst_count, dst_height, dst_width = destination.shape
            else:
                dst_count = 1
                dst_height, dst_width = destination.shape
        
        
        dst_bounds = array_bounds(dst_height, dst_width, dst_transform)
    
    
    if crop:
        return clip_array_crop(source, destination, src_bounds, dst_bounds,
                               src_crs=src_crs,src_nodata=src_nodata,
                               dst_crs=dst_crs,dst_nodata=dst_nodata,
                               nodata=nodata,
                               mode=mode,
                               masked=masked,
                               crop_use_index=crop_use_index,
                               with_complement=with_complement,
                               stacklevel=stacklevel+1
                               )
    else:
        return clip_array(source, src_bounds, dst_bounds,
                          src_crs=src_crs, src_nodata=None,
                          dst_crs=dst_crs,
                          nodata=nodata,
                          mode=mode,
                          masked=masked,
                          with_complement=with_complement,
                          stacklevel=stacklevel+1)









