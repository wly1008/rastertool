# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:12:39 2024

@author: wly
"""

import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.crs import CRS
from rasterio.enums import MaskFlags
from rasterio.windows import Window
from rasterio.warp import transform_bounds

import warnings
import numpy as np
import logging, pathlib, os
from contextlib import ExitStack

from mycode.arcmap import get_RasterAttr
from mycode.rio_wrap.core import reproject as _reproject
from mycode.rio_wrap.warnings import SetNodataWarning, SetRasterAttrWarning

logger = logging.getLogger(__name__)

def create_raster(**kwargs):
    memfile = rasterio.MemoryFile()
    return memfile.open(**kwargs)

def align(array, pad_width, mode='constant', **kwargs):
    '''在np.pad基础上加入负值处理,
       pad_width中负值代表剔除,正值代表填充
       参数说明见np.pad
    '''
    pad_width = np.asarray(pad_width)
    
    # 替换负值运行np.pad
    _pad_width = pad_width.copy()
    _pad_width[_pad_width<0] == 0
    
    ret0 = np.pad(array, _pad_width, mode=mode, **kwargs)
    
    # 将负值理解为剔除，计算切片
    slices = []
    
    r = pad_width[:,0]
    l = pad_width[:,1]
    shape = ret0.shape
    for dim in range(ret0.ndim):
        if r[dim] < 0 :
            r[dim] = -r[dim]
        if l[dim] >= 0:
            l[dim] = shape[dim]
        
        slices.append(slice(r[dim], l[dim]))
        
    return ret0[tuple(slices)]

def arr_off(arr_dst, src_transform, dst_transform, nodate=0):
    '''
    偏移纠正
    偏移一个以内，同分辨率
    暂只用与掩膜纠正
    
    Parameters
    ----------
    arr_dst : TYPE
        掩膜数组.
    src_transform : TYPE
        被掩膜transform.
    dst_transform : TYPE
        掩膜transform.
    nodate : TYPE, optional
        掩膜数组填充值. The default is 0.

    Returns
    -------
    TYPE
        纠正后的掩膜数组.

    '''
    src_xres, _, src_left, _, src_yres, src_top, _, _, _ = src_transform
    src_yres = -src_yres
    
    dst_xres, _, dst_left, _, dst_yres, dst_top, _, _, _ = dst_transform
    dst_yres = -dst_yres
    
    

    if arr_dst.ndim == 2:
        arr_dst = np.array([arr_dst])
        
    shape = arr_dst.shape
    count, height, width = shape
    
    # numx, modx = divmod(dst_left - src_left, src_xres)
    
    if dst_left - src_left > 0.5 * src_xres:
        col = [1, 0]
        coff = 0
    
    elif dst_left - src_left < -0.5 * src_xres:
        col = [0, 1]
        coff = 1
    
    else:
        col = [0, 0]
        coff = 0
    
    if dst_top - src_top > 0.5 * src_yres:
        row = [0, 1]
        roff = 1
    elif dst_top - src_top < -0.5 * src_yres:
        
        row = [1, 0]
        roff = 0
    else:
        row = [0, 0]
        roff = 0
    # print(row, col)
    return np.pad(arr_dst,[[0, 0], row, col], mode='constant', constant_values=nodate)[:, roff : height+roff, coff : width+coff]



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

def round0(x):
    if x % 1 == 0.5:
        return np.ceil(x)
    else:
        return np.round(x)
# 窗口取整函数
def round_window(win):
    '''四舍五入'''
    col_off, row_off, width, height = win.col_off, win.row_off, win.width, win.height
    
    col_off, width = tr_winattr(col_off, width, round0)
    row_off, height = tr_winattr(row_off, height, round0)
    
    
    # col_off, row_off, width, height = np.round((win.col_off, win.row_off, win.width, win.height)).astype(int)
    return Window(col_off, row_off, width, height)



def rasterio_window(win):
    '''rio.clip标准'''
    
    out_window = win.round_lengths()  # int(math.floor(x + 0.5)) 四舍五入
    out_window = out_window.round_offsets()  # int(math.floor(x + 0.001)) 近乎向下取整，在0.001范围向上取
    
    return out_window



def touch_window(win):
    '''保留所有接触像元'''
    col_off, row_off, width, height = win.col_off, win.row_off, win.width, win.height
    
    col_off, width = tr_winattr(col_off, width, *(int, np.ceil))
    row_off, height = tr_winattr(row_off, height, *(int, np.ceil))
    
    
    return Window(col_off, row_off, width, height)

def touch_arr(arr_crop,src_transform, dst_transform, nodate=0):
    
    arr_right = np.pad(arr_crop,[[0, 0], [0, 1], [1, 0]],
                       mode='constant', constant_values=0)
    arr_down = np.pad(arr_crop,[[0, 0], [1, 0], [0, 1]],
                       mode='constant', constant_values=0)
    
    arr_crop = arr_crop | arr_down | arr_right
    arr_crop[:,-1,-1] = arr_crop[:,-2,-2]
    return arr_crop

def _unify(raster_in, dst_in=None, out_path=None,
         get_ds=True,
         bounds=None, 
         mode='round',
         nodata=None,
         dtype=None,
         projection='geographic',
         with_complement=True,
         crop=False,
         arr_crop=None,
         crop_use_index=None,
         dst_transform=None,
         unify=0, 
         unify_options=None,
         delete=False,
         stacklevel=3,
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
    projection : CRS, optional
        输入范围的空间参考. The default is 'geographic'.
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
    unify : int, optional
        统一参数集. The default is 0.
        0 {}
        1 {'ushape':True, 'csize':True, 'tolerance':0.001}
        2 {'resample':True, ushape':True}
        3 {'resample':True, 'double_operation':True, 'ushape':True}
    unify_options : dict, optional
        统一参数
        {'ushape':True, 'csize':True, 'tolerance':0.001, 'resample':True, 'reproject':True,'double_operation':True,}
        ushape: 是否统一shape, Default: False.
        csize: 是否检查分辨率是否统一, Default: False.
        tolerance: size检查的允许容差(相对值). Default: 0.
        resample、reproject: 是否重采样、重投影 . Default: False.
        double_operation: 是否预裁剪((原数据远大于目标范围时建议使用))  Default: False.
    delete : TYPE, optional
        是否删除输入栅格raster_in（清除中间变量）,
        当raster_in为地址时正常执行, 而输入栅格类变量时不会, 除非使用'!True'.
        The default is False.
    stacklevel : int opt
        警告级别，默认为3，详见warnings.warn
    **creation_options : TYPE
        输出栅格其他profile更新选项


    Raises
    ------
    ValueError
        DESCRIPTION.
    click
        DESCRIPTION.

    Returns
    -------
    if out_path:生成栅格文件，返回文件地址
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重投影后的栅格矩阵（array）和 profile
    
    '''
    
    # 获取值
    def get_value(key):
        if dst_in:
            value = get_RasterAttr(dst_in, key)
        else:
            
            try:
                value = unify_kwargs[key]
            except:
                raise ValueError(f'Plase input "{key}" to "unify_options" or input "dst_in"')
        return value
    
    
    with ExitStack() as stack:
        
        # 
        src = stack.enter_context(rasterio.open(raster_in)) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
        
        # unify设置
        dic_unify = {0:{},
                     1:{'ushape': True, 'csize': True, 'tolerance': 0.001},
                     2:{'resample': True, 'ushape': True, 'how':0},
                     3:{'resample': True, 'double_operation': True, 'ushape': True, 'how': 0}
                     
                    }
        
        # unify相关参数读取
        unify_options = unify_options if isinstance(unify_options, dict) else {}
        unify_kwargs = dic_unify[unify]
        unify_kwargs.update(**unify_options)
        
        
        reproject = unify_kwargs.get('reproject', False)
        resample = unify_kwargs.get('unify_kwargs', False)
        
        how = unify_kwargs.get('how', 0)
        double_operation = unify_kwargs.get('double_operation', False)
        
        ushape = unify_kwargs.get('ushape', False)
        csize = unify_kwargs.get('csize', False)
        tolerance = unify_kwargs.get('tolerance', 0)
        
        
        
        # 裁剪前操作(重投影、重采样)
        if resample:
            
            if double_operation:
                src = _unify(src,
                           dst_in=dst_in,bounds=bounds,get_ds=True,
                           nodata=nodata,mode=mode,
                           projection=projection,with_complement=with_complement,
                           **creation_options)
            
            size = get_value("size")
            if reproject:
                crs = get_value('crs')
            else:
                crs = None
            src = _reproject(src, crs=crs, how=how, resolution=size)
            
        else:
            if reproject:
                if double_operation:
                    src = _unify(src,
                               dst_in=dst_in,bounds=bounds,get_ds=True,
                               projection=projection,with_complement=with_complement,
                               **creation_options)
                crs = get_value('crs')
                if crs != src.crs:
                    src = _reproject(src, crs=crs, how=how)
                else:
                    pass
            
        
        
        
        
        if not src.transform.is_rectilinear:
            raise ValueError(
                "Non-rectilinear rasters (i.e. with rotation or shear) cannot be clipped"
            )
    


        if dst_in:
            # with rasterio.open(dst_in) as template_ds:
            template_ds = stack.enter_context(rasterio.open(dst_in)) if issubclass(type(dst_in), (str,pathlib.PurePath)) else dst_in
            bounds = template_ds.bounds
            if template_ds.crs != src.crs:
                bounds = transform_bounds(template_ds.crs, src.crs,
                                          *bounds)
        elif bounds:
            if projection == 'geographic':
                bounds = transform_bounds(CRS.from_epsg(4326), src.crs, *bounds)
            elif issubclass(type(projection), CRS):
                bounds = transform_bounds(projection, src.crs, *bounds)
            
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
    
    
        
        
    
        if csize:
            
            size = get_value('size')
            src_size = get_RasterAttr(src,'xsize')
            if tolerance:
                tolerance = abs(tolerance)
                run = abs(src_size - size) <= src_size*tolerance
            else:
                run = (src_size == size)
        else:
            run = True
        
        if ushape and run:
            
            # template_ds.closed
            if dst_in:
                shape = template_ds.shape
            else:
                
                try:
                    shape = unify_kwargs['shape']
                except:
                    raise ValueError('Plase input "shape" to "unify_options" or input "dst_in"')
            height, width = shape
            
            
            bounds_window = Window(bounds_window.col_off, bounds_window.row_off, width, height)
            
            

            

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
    
        height = int(out_window.height)
        width = int(out_window.width)
    
        out_kwargs = src.profile
        transform = src.window_transform(out_window)
        src_nodata = src.nodata
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
        # np.unique(arr)
        # masked = creation_options.get('masked',True)
        arr = src.read(
                        window=out_window,
                        out_shape=(src.count, height, width),
                        boundless=True,
                        masked=True,
                        )

        if MaskFlags.per_dataset in src.mask_flag_enums[0]:
            per_dataset = True
            arr_mask = src.read_masks(
                                      window=out_window,
                                      out_shape=(src.count, height, width),
                                      boundless=True,
                                      )[0]
        else:
            per_dataset = False
        if crop:
            if crop_use_index is None:
                crop_use_slice = slice(None)
            elif isinstance(crop_use_index, int):
                crop_use_slice = slice(crop_use_index, crop_use_index+1)
            elif isinstance(crop_use_index, slice):
                crop_use_slice = crop_use_index
            else:
                raise TypeError(f'--crop_use_index is {type(crop_use_index)},but it must be an int or a slice.')
            
            win_right = round_window(bounds_window)
            # src_transform = transform
            if dst_in:
                
                # arr_crop = template_ds.read_masks(out_shape=arr.shape)
                arr_crop = template_ds.read_masks(out_shape=(template_ds.count, win_right.width, win_right.height))
                
            
            elif arr_crop is None:
                raise ValueError('--arr_crop or --dst_in required')
            
            if arr_crop.ndim == 2:
                arr_crop = np.asarray([arr_crop])
            arr_crop = arr_crop[crop_use_slice]
            if not ((arr_crop.shape[0] == 1) or (arr_crop.shape[0] == arr.shape[0])):
                raise ValueError('arr_crop 与 arr波段数不匹配且arr_crop不为单波段, 无法广播, 可尝试使用crop_use_index参数')
            
            # 计算调整arr_crop, 使与arr对齐
            if arr_crop.shape[1:] != arr.shape[1:]:
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
                
            
            arr = np.where(arr_crop == 0, src_nodata, arr)

            if per_dataset:
                arr_mask = np.where(arr_crop[0] == 0, False, arr_mask)
                # arr = np.array([[1,2],[1,2],[1,2]])
                # arr_crop = np.array([[1,0],[1,0]])
                # np.where(arr_crop==0, 9, arr)
    # y = np.array([[1,2],[1,3]])
    # x = y[slice(3)]
    
    # 删除中间栅格
    if delete and issubclass(type(raster_in), (str,pathlib.PurePath)):
        
        os.remove(raster_in)
    elif delete == '!True':
        file = raster_in.files[0]
        raster_in.close()
        os.remove(file)
    
    
    
    # 更新无效值
    if nodata is None:
        if out_kwargs['nodata'] is None:
            out_kwargs['nodata'] = 0
            arr = np.where(arr == src_nodata, out_kwargs['nodata'], arr)
            warnings.warn(
                "源数据无nodata, 自动设定为0, 注意可能产生的影响。 如非所需请为参数nodata赋值",
                SetNodataWarning,
                stacklevel=stacklevel
                            )
    else:
        out_kwargs['nodata'] = nodata
        if src_nodata is None:
            # np.isnan(None) 会报错
            arr  = np.where(arr == src_nodata, out_kwargs['nodata'], arr)
        elif np.isnan(src_nodata):
            arr  = np.where(np.isnan(arr), out_kwargs['nodata'], arr)
        else:
            arr = np.where(arr == src_nodata, out_kwargs['nodata'], arr)


    
    if dtype is not None:
        out_kwargs['dtype'] = dtype
    

    
    # 输出
    try:
        if out_path:
            with rasterio.open(out_path, "w", **out_kwargs) as out:
                out.write(arr)
                out.descriptions = bandNames
                if per_dataset:
                    out.write_mask(arr_mask)
            return out_path
        elif get_ds:
            out = create_raster(**out_kwargs)
            out.write(arr)
            out.descriptions = bandNames
            if per_dataset:
                out.write_mask(arr_mask)
            return out
        else:
            return (arr, out_kwargs)
    except (TypeError, ValueError) as e:
        args = e.args
        args = args + ('可能是nodata与栅格dtype不匹配, 可用nodata、dtype参数设置转换',)
        raise TypeError(*args)
    except Exception as e:
        raise e



if __name__ == '__main__':
    raster_in = r'F:/PyCharm/pythonProject1/arcmap/015温度/土地利用/landuse_4y/1990-5km-tiff.tif'

    dst_in = r'F:\PyCharm\pythonProject1\arcmap\007那曲市\data\eva平均\eva_2.tif'

    out_path = r'F:\PyCharm\pythonProject1\代码\mycode\测试文件\eva5_1.tif'
    
    out_path1 = r'F:\PyCharm\pythonProject1\arcmap\015温度\zonal\grand_average.xlsx'
    
    src = rasterio.open(dst_in)
    src.profile
    arr_mask = src.read_masks()
    a1 = src.dataset_mask()
    arr = src.read(1)
    
    x= a1 == arr_mask
    np.unique(x)



































