# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 10:44:58 2026

@author: wly
"""



import os
import numpy as np
import pandas as pd
import rasterio.merge


from rastertool.functions import get_dataset_opener, read, set_nodata, output, out

from scipy.ndimage import convolve,percentile_filter


def create_kernel(width, deleted=False):
    '''
    创建邻域卷积核，可选择是否去除中心像元（用于 focal 统计）

    Parameters
    ----------
    width : TYPE
        DESCRIPTION.

    Returns
    -------
    kernel : TYPE
        DESCRIPTION.

    '''
    win_size = width  # 邻域窗口大小
    kernel = np.ones((win_size, win_size), dtype=np.float32)  # 先创建全True的21×21矩阵
    if deleted:
        if width == 1:
            raise ValueError("width=1 时无法删除中心像元")
        center_idx = win_size // 2  # 计算中心索引：21//2 = 10
        kernel[center_idx, center_idx] = 0  # 核心：把中心像元置为False，剔除中心像元            
    return kernel



def focal_mean(source, radius, out_path, deleted=False, mode='reflect', crop=True,
               nodata=None, dtype=None, compress=None, update_stats=False):
    """
    局部邻域均值（Focal Mean）计算函数，支持忽略 NoData，
    并可选择是否剔除中心像元，以及是否裁剪原始 NoData 区域。

    Parameters
    ----------
    source : str or dataset
        输入栅格数据路径或已打开的数据集对象。
    radius : int
        邻域窗口半径（width = radius + 1 + radius,如 3 表示 7×7，5 表示 11×11）。
    out_path : str or None
        输出栅格路径。
        若为 None，则不写文件，直接返回计算结果和 profile。
    deleted : bool, optional
        是否剔除中心像元：
        - False：中心像元参与均值计算（默认）
        - True ：中心像元不参与均值计算
    mode : str, optional
        边界扩展方式，传递给 ``scipy.ndimage.convolve``，
        常用值包括：
        'reflect'（默认）、'nearest'、'constant'、'mirror'、'wrap'。
    crop : bool, optional
        是否裁剪原始 NoData 区域。
        - True（默认）：输出结果在原始 NoData 像元位置仍为 NoData，
          即只对原始有效像元位置输出均值结果。
        - False：只要邻域内存在有效像元，就计算均值；
          仅当邻域内完全没有有效像元时，结果才为 NoData。
    nodata : optional
        输出栅格的 NoData 值。
        若为 None，则沿用输入数据的 NoData。
    dtype : optional
        输出栅格的数据类型。
        若为 None，则沿用输入数据类型。
    compress : optional
        输出栅格的压缩方式（如 'lzw'）。
        若为 None，则沿用输入 profile 中的设置。
    update_stats : bool, optional
        写出文件时是否更新统计信息。

    Returns
    -------
    dest : ndarray
        计算得到的局部均值数组（当 out_path 为 None 时返回）。
    profile : dict
        更新后的栅格 profile（当 out_path 为 None 时返回）。

    Notes
    -----
    - 本函数采用“mask + 卷积”的方式计算邻域均值：
      仅对有效像元参与统计。
    - ``crop`` 参数用于控制输出结果是否严格受原始 NoData 掩膜约束：
      * ``crop=True`` 适用于保持原始数据覆盖范围不变的场景；
      * ``crop=False`` 适用于希望对 NoData 边缘进行邻域扩展计算的场景。
    - 当邻域内不存在任何有效像元时，输出结果为 NoData。
    """

    
    # 1. 构建邻域卷积核（全 1 矩阵，可选去中心像元）
    width = radius + 1 + radius
    kernel = create_kernel(width, deleted)

    
    # 2. 读取栅格数据（masked=True 表示 NoData 会进入 mask）
    data, profile = read(source, masked=True)

    
    # 3. 构造有效像元掩膜和数值数组
    #    valid_mask : 有效像元为 1，无效为 0
    #    data_values: NoData 填 0，不影响后续求和
    valid_mask = (~data.mask).astype(np.uint8)
    data_values = data.filled(0)

    
    # 4. 卷积统计
    #    sum_valid : 邻域内有效像元数量
    #    sum_value : 邻域内有效像元值的总和
    sum_valid = convolve(valid_mask, kernel, mode=mode)
    sum_value = convolve(data_values, kernel, mode=mode)

    
    # 5. 计算局部均值
    #    使用 np.maximum(sum_valid, 1) 防止除以 0
    mean_value = sum_value / np.maximum(sum_valid, 1)

    
    
    if crop:  # 使用源数据掩膜
        mask = data.mask
    else:  # 当邻域内没有任何有效像元时，标记为 NoData
        mask = sum_valid == 0

    
    # 6. 处理输出的 NoData、数据类型与 profile
    output(out_path, mean_value, mask, profile, nodata, dtype, compress, update_stats)
    
    

    


from scipy.ndimage import maximum_filter, minimum_filter

def _focal_nonlinear(values, valid_mask, kernel, stat, mode, cval=0.0):

    footprint = kernel.astype(bool)
    
    dtype = values.dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    if stat == 'max':
        fill = info.min
        func = maximum_filter
    else:
        fill = info.max
        func = minimum_filter

    arr = np.where(valid_mask, values, fill)
    return func(arr, footprint=footprint, mode=mode, cval=cval)


def _focal_var_fast(values, valid_mask, kernel, mode, cval=0.0, count=None):
    """
    高性能 focal var（O(1) 滑窗）
    """

    # 只在 valid 区域保留值
    v = np.where(valid_mask, values, 0.0)

    # 计数
    if count is None:
        count = convolve(valid_mask, kernel, mode=mode, cval=cval)

    # 一阶、二阶矩
    sum_x = convolve(v, kernel, mode=mode, cval=cval)
    sum_x2 = convolve(v * v, kernel, mode=mode, cval=cval)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean = sum_x / count
        var = sum_x2 / count - mean ** 2

    # 数值安全（浮点误差可能导致负数）
    var = np.maximum(var, 0)
    
    return var


    
def _focal_perc_fast(values, valid_mask, kernel, mode,
                     cval=0.0, count=None, q=50):

    footprint = kernel.astype(bool)

    dtype = values.dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)

    fill = info.min

    # 1. 无效值 → 最小值
    arr = np.where(valid_mask, values, fill)

    # 2. 计数
    if count is None:
        count = convolve(valid_mask, kernel, mode=mode, cval=cval)

    # 3. 等效分位数修正
    num = kernel.sum()
    qx = 100 * (count * q / 100 + (num - count)) / num

    # 4. focal percentile
    result = percentile_filter(
        arr,
        percentile=qx,
        footprint=footprint,
        mode=mode,
        cval=fill
    )

    return result
    
    
    
def _focal_max(values, valid_mask, kernel, mode, cval=0.0):
    return _focal_nonlinear(values, valid_mask, kernel, stat='max', mode=mode, cval=cval)
    
    

def _focal_min(values, valid_mask, kernel, mode, cval=0.0):
    return _focal_nonlinear(values, valid_mask, kernel, stat='mix', mode=mode, cval=cval)


def _focal_sum(values, valid_mask, kernel, mode, cval=0.0):
    return convolve(values, kernel, mode=mode, cval=cval)


def _focal_mean(values, valid_mask, kernel, mode, cval=0.0, count=None):
    if count is None:
        count = convolve(valid_mask, kernel, mode=mode, cval=cval)
    return convolve(values, kernel, mode=mode, cval=cval) / count

def _focal_count(values, valid_mask, kernel, mode, cval=0.0, count=None):
    if count is None:
        count = convolve(valid_mask, kernel, mode=mode, cval=cval)
    
    return count

def _focal_var(values, valid_mask, kernel, mode, cval=0.0, count=None):
    return _focal_var_fast(values, valid_mask, kernel, mode, cval=cval, count=count)

def _focal_std(values, valid_mask, kernel, mode, cval=0.0, count=None, var=None):
    if var is None:
        var = _focal_var_fast(values, valid_mask, kernel, mode, cval=cval, count=count)
    
    return np.sqrt(var)


def _focal_perc(values, valid_mask, kernel, mode, cval=0.0, count=None, q=50):
    
    return _focal_perc_fast(values, valid_mask, kernel, mode, cval=cval, count=count, q=q)


def _focal_median(values, valid_mask, kernel, mode, cval=0.0, count=None):
    return _focal_perc_fast(values, valid_mask, kernel, mode, cval=cval, count=count, q=50)

    
def focal_tool(source, radius, stat,
               out_path=None,
               deleted=False,
               q=50,
               mode='reflect',
               cval=0.0,
               crop=True,
               nodata=None,
               dtype=None,
               compress=None,
               update_stats=False):
    """
    通用 focal 统计框架
    stat: 'sum' | 'mean' | 'max' | 'min' | 'count' | 'std' | 'var'
    """

    # ---------- 1. kernel ----------
    width = radius * 2 + 1
    kernel = create_kernel(width, deleted)

    # ---------- 2. read data ----------
    data, profile = read(source, masked=True)

    valid_mask = ~data.mask
    values = data.filled(0)
    
    
    sum_valid = convolve(valid_mask, kernel, mode=mode, cval=cval)
    # ---------- 3. 统计 ----------
    if stat in ('sum', 'mean'):
        
        sum_value = convolve(values, kernel, mode=mode, cval=cval)

        if stat == 'sum':
            result = sum_value
        else:
            result = sum_value / np.maximum(sum_valid, 1)

    elif stat in ('max', 'min'):
        result = _focal_nonlinear(
            values, valid_mask, kernel, stat, mode, cval
        )
    elif stat in('std', 'var'):
        var_value, _ = _focal_var_fast(
            values, valid_mask, kernel, mode, cval, sum_valid
        )
        
        if stat == 'std':
            result = np.sqrt(var_value)
        else:
            result = var_value
    elif stat in ('perc', 'median'):
        if stat == 'median':
            q = 50
        
        result = _focal_perc_fast(values, valid_mask, kernel, mode, cval, sum_valid, q)
        

    elif stat == 'count':
        result = sum_valid
        

    else:
        raise ValueError(f"Unsupported stat: {stat}")

    # ---------- 4. mask ----------
    if crop:
        mask = data.mask
    else:
        mask = sum_valid == 0

    # ---------- 5. output ----------
    return output(
        out_path, result, mask, profile,
        nodata, dtype, compress, update_stats
    )

    
    
    














