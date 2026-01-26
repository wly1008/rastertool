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

import math
import numpy as np
import logging, pathlib, os
from contextlib import ExitStack


from mycode.arcmap import create_raster, get_RasterAttr
from mycode.arcmap import reproject as _reproject



logger = logging.getLogger(__name__)

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



def tr_winattr(off, length,*funcs,axis='col'):
    '''调用取整方法，并消除off改变对length+off的影响'''
    off_tr = funcs[0](off)
    
    if axis == 'col':
        length_tr = off + length - off_tr
    elif axis == 'row':
        length_tr = off_tr - (off - length)
    length_tr = funcs[-1](length_tr)
    return int(off_tr), int(length_tr)

# 窗口取整函数
def round_window(win):
    '''四舍五入'''
    col_off, row_off, width, height = win.col_off, win.row_off, win.width, win.height
    
    col_off, width = tr_winattr(col_off, width, round, axis='col')
    row_off, height = tr_winattr(row_off, height, round, axis='row')
    
    
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
    
    col_off, width = tr_winattr(col_off, width, *(int, np.ceil), axis='col')
    row_off, height = tr_winattr(row_off, height, *(int, np.ceil), axis='row')
    
    
    return Window(col_off, row_off, width, height)



def clip(raster_in, dst_in=None, out_path=None,
         get_ds=True,
         bounds=None, 
         mode='round',
         nodata = 'None',
         projection='geographic',
         with_complement=True,
         crop=False,
         arr_crop=None,
         # dst_transform=None,
         unify=0, 
         unify_options=None,
         delete=False,
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
        输出栅格无效值,为字符串"None"时源栅格一致. The default is 'None'.
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
        DESCRIPTION. The default is False.
    **creation_options : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

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
        
        # crop = unify_kwargs.get('crop',False)
        
        
        # 裁剪前操作(重投影、重采样)
        if resample:
            
            if double_operation:
                src = clip(src,
                           dst_in=dst_in,bounds=bounds,get_ds=True,
                           projection=projection,with_complement=with_complement,
                           **creation_options)
            
            size = get_value("size")
            if reproject:
                crs = get_value('crs')
            else:
                crs = 'src'
            src = _reproject(src, crs=crs, how=how, resolution=size)
            
        else:
            if reproject:
                if double_operation:
                    src = clip(src,
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
        
        
        
        out_kwargs.update({
            'height': height,
            'width': width,
            'transform': transform})
        
        out_kwargs.update(**creation_options)
        
        if "blockxsize" in out_kwargs and int(out_kwargs["blockxsize"]) > width:
            del out_kwargs["blockxsize"]
            logger.warning(
                "Blockxsize removed from creation options to accomodate small out_path width"
            )
        if "blockysize" in out_kwargs and int(out_kwargs["blockysize"]) > height:
            del out_kwargs["blockysize"]
            logger.warning(
                "Blockysize removed from creation options to accomodate small out_path height"
            )
        # np.unique(arr)
        masked = creation_options.get('masked',True)
        arr = src.read(
                        window=out_window,
                        out_shape=(src.count, height, width),
                        boundless=True,
                        masked=masked,
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
            # src_transform = transform
            if dst_in:
                
                arr_crop = template_ds.read_masks(out_shape=arr.shape)
                
            #     dst_transform = template_ds.transform
                
                
            #     arr_crop = arr_off(arr_crop, src_transform, dst_transform)
            # elif dst_transform:
            #     arr_crop = arr_off(arr_crop, src_transform, dst_transform)
            
            elif arr_crop is None:
                raise ValueError('--arr_crop or --dst_in required')
            
            
            
            arr = np.where(arr_crop == 0, src_nodata, arr)
            if per_dataset:
                arr_mask = np.where(arr_crop[0] == 0, src_nodata, arr_mask)
            
    
    
    # 删除中间栅格
    if delete and issubclass(type(raster_in), (str,pathlib.PurePath)):
        
        os.remove(raster_in)
    elif delete == '!True':
        file = raster_in.files[0]
        raster_in.close()
        os.remove(file)
    
    
    # 更新无效值
    if nodata != 'None':
        out_kwargs['nodata'] = nodata
        if src_nodata is None:
            arr  = np.where(arr == src_nodata, nodata, arr)
        elif np.isnan(src_nodata):
            arr  = np.where(np.isnan(arr), nodata, arr)
        else:
            arr = np.where(arr == src_nodata, nodata, arr)
    # np.unique(arr.astype(float))
    
    # 输出
    if out_path:
        with rasterio.open(out_path, "w", **out_kwargs) as out:
            out.write(arr)
    
            if per_dataset:
                out.write_mask(arr_mask)
        return out_path
    elif get_ds:
        out = create_raster(**out_kwargs)
        out.write(arr)
        if per_dataset:
            out.write_mask(arr_mask)
        return out
    else:
        return (arr, out_kwargs)



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



































