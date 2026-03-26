# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 10:44:58 2026

@author: wly
"""



import numpy as np

# from rastertool.core._focaltool import percentile_filter_array_q
from rastertool.functions import read, output

from scipy.ndimage import convolve,percentile_filter


from scipy.ndimage import maximum_filter, minimum_filter,generic_filter


FOCAL_STATS = {}
def register_focal(name):
    def decorator(func):
        FOCAL_STATS[name] = func
        return func
    return decorator


def Rectangle_kernel(width, deleted=False):
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
    kernel = np.ones((width, width), dtype=np.float32)  # 先创建全True的21×21矩阵
    if deleted:
        if width == 1:
            raise ValueError("width=1 时无法删除中心像元")
        center_idx = width // 2  # 计算中心索引：21//2 = 10
        kernel[center_idx, center_idx] = 0  # 核心：把中心像元置为False，剔除中心像元            
    return kernel



def Round_kernel(width, deleted=False):
    '''
    创建圆形邻域卷积核，可选择是否去除中心像元

    Parameters
    ----------
    width : int
        核大小（必须为奇数）
    deleted : bool
        是否删除中心像元

    Returns
    -------
    kernel : np.ndarray
        圆形卷积核
    '''
    
    if width % 2 == 0:
        raise ValueError("width 必须为奇数")

    radius = width // 2

    # 创建坐标网格
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

    # 圆形掩膜（关键）
    mask = x**2 + y**2 <= radius**2

    kernel = mask.astype(np.float32)

    # 是否删除中心像元
    if deleted:
        if width == 1:
            raise ValueError("width=1 时无法删除中心像元")
        kernel[radius, radius] = 0

    return kernel



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
    
    return var, count


    
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
    # qx = q

    # 4. focal percentile
    result = percentile_filter(
        arr,
        percentile=qx,
        footprint=footprint,
        mode=mode,
        cval=fill
    )

    return result, count

    

def is_num_dtype(values, dtype='float64'):
    try:
        return np.dtype(values.dtype) == np.dtype(dtype)
    except Exception:
        return False
from functools import partial
def _focal_perc(values, valid_mask, kernel, mode,
                     cval=0.0, q=50):
    footprint = kernel.astype(bool)
    percentile = partial(np.percentile, q=q, method='nearest')
    
    if np.dtype(values.dtype).kind != "f":
        
        values = np.asarray(values, dtype='float32')
        
    
    arr = np.where(valid_mask, values, np.nan)
    
    def func(window):
        vals = window[~np.isnan(window)]
        if vals.size == 0:
            return np.nan
        
        return percentile(vals)
    
    
    result = generic_filter(
        arr,
        function=func,
        footprint=footprint,
        mode=mode,
        cval=np.nan,
    )
    
    
    
    return result

        
        
    
    
    
    
    
    



@register_focal("max")
def focal_max(values, valid_mask, kernel, *, mode, cval, cache):
    
    if 'max' not in cache:
        
        cache['max'] = _focal_nonlinear(values, valid_mask, kernel, stat='max', mode=mode, cval=cval)
    return cache['max'] 
    
@register_focal("min")
def focal_min(values, valid_mask, kernel, *, mode, cval, cache):
    
    if 'min' not in cache:
        
        cache['min'] = _focal_nonlinear(values, valid_mask, kernel, stat='min', mode=mode, cval=cval)
    return cache['min'] 

@register_focal("sum")
def focal_sum(values, valid_mask, kernel, *, mode, cval, cache):
    
    if 'sum' not in cache:
        
        cache["sum"] = convolve(values, kernel, mode=mode, cval=cval)
    
    return cache["sum"]
        
        

@register_focal("mean")
def focal_mean(values, valid_mask, kernel, *, mode, cval, cache):
    
    if 'mean' not in cache:
        count = focal_count(values, valid_mask, kernel, mode=mode, cval=cval, cache=cache)
        sum = focal_sum(values, valid_mask, kernel, mode=mode, cval=cval, cache=cache)
        cache["mean"] = sum / np.maximum(count, 1)
        
    return cache["mean"]


@register_focal("count")
def focal_count(values, valid_mask, kernel, *, mode, cval, cache):
    if "count" not in cache:
        cache["count"] = convolve(valid_mask.astype('uint8'), kernel, mode=mode, cval=cval)
    return cache["count"]


@register_focal("var")
def focal_var(values, valid_mask, kernel, *, mode, cval, cache):
    if "var" not in cache:
        count = focal_count(values, valid_mask, kernel, mode=mode, cval=cval, cache=cache)
        cache["var"], _ = _focal_var_fast(
            values, valid_mask, kernel, mode, cval, count
        )
    return cache["var"]


@register_focal("std")
def focal_std(values, valid_mask, kernel, *, mode, cval, cache):
    
    if 'std' not in cache:
        var = focal_var(
            values, valid_mask, kernel, mode=mode, cval=cval, cache=cache
        )
        cache["std"] = np.sqrt(var)
    
    return cache["std"]
        


@register_focal("perc")
def focal_perc(values, valid_mask, kernel, *, mode, cval, cache, q=50):
    key = f"perc_{q}"
    if key not in cache:
        cache[key] =  _focal_perc(values, valid_mask, kernel, mode, cval, q)   
        # count = focal_count(values, valid_mask, kernel, mode=mode, cval=cval, cache=cache)
        # cache[key],_ =  _focal_perc_fast(values, valid_mask, kernel, mode, cval, count, q)
        
    
    return cache[key]


@register_focal("median")
def focal_median(values, valid_mask, kernel, *, mode, cval, cache):
    if "perc_50" not in cache:
        
        
        
        cache["perc_50"]=  _focal_perc(values, valid_mask, kernel, mode, cval, q=50)
        # count = focal_count(values, valid_mask, kernel, mode=mode, cval=cval, cache=cache)
        # cache["perc_50"], _ =  _focal_perc_fast(values, valid_mask, kernel, mode, cval,count, q=50)
        
    return cache["perc_50"]



def focaltool(
    source,
    radius,
    stat,
    *,
    out_path=None,
    deleted=False,
    q=50,
    Round=True,
    mode="reflect",
    cval=0.0,
    crop=True,
    nodata=None,
    dtype=None,
    compress=None,
    update_stats=False,
    **kwargs,
):
    
    """
    对输入栅格执行 focal（邻域）统计运算。

    以每个像元为中心，在给定半径的邻域窗口内，
    对所有有效像元（非 NoData）计算指定的统计量。

    Parameters
    ----------
    source : str or raster-like
        输入栅格数据源。可以是文件路径，或 rastertool 支持的数据集对象。
        读取时将以 masked=True 的方式加载，NoData 会被自动识别。

    radius : int
        邻域半径（像元单位）。
        实际窗口大小为：
            (2 * radius + 1) × (2 * radius + 1)

    stat : str or callable
        要计算的邻域统计量。
        
        - str：内置统计名称之一：
            - "count"   : 有效像元数量
            - "sum"     : 邻域和
            - "mean"    : 邻域均值
            - "min"     : 邻域最小值
            - "max"     : 邻域最大值
            - "var"     : 邻域方差（总体方差）
            - "std"     : 邻域标准差
            - "perc"    : 邻域分位数（由参数 q 指定）
            - "median"  : 邻域中位数（等价于 perc, q=50）

        - callable：
            自定义统计函数，签名需与内置 focal 函数一致：
            func(values, valid_mask, kernel, *, mode, cval, cache, **kwargs)

    out_path : str, optional
        输出栅格路径。
        - 若为 None，则返回内存结果
        - 若提供路径，则写出为栅格文件

    deleted : bool, default False
        是否在邻域计算中删除中心像元。
        
        - False：中心像元参与统计（默认）
        - True ：中心像元不参与统计（常用于邻域对比、空间自相关分析）

    q : int or float, default 50
        分位数参数，仅在 stat="perc" 或使用分位数统计时有效。
        取值范围通常为 [0, 100]。

    mode : str, default "reflect"
        边界处理方式，直接传递给 scipy.ndimage 的卷积/滤波函数。
        常见取值：
            - "reflect"
            - "nearest"
            - "constant"
            - "mirror"
            - "wrap"

    cval : float, default 0.0
        当 mode="constant" 时，边界填充值。

    crop : bool, default True
        输出 mask（NoData 区域）控制方式：

        - True：
            输出栅格的 mask 与输入栅格完全一致，
            不因邻域中无有效像元而扩展 NoData 区域。

        - False：
            若某像元邻域内无任何有效值（count == 0），
            则该像元输出为 NoData。

    nodata : number, optional
        输出栅格的 NoData 值。
        若为 None，则沿用输入栅格或 profile 中的设置。

    dtype : numpy dtype, optional
        输出栅格的数据类型。
        若为 None，则自动推断或沿用输入类型。

    compress : str or dict, optional
        输出栅格的压缩方式，例如：
            - "lzw"
            - "deflate"
        或 rasterio 风格的压缩参数字典。

    update_stats : bool, default False
        是否在写出栅格时更新统计信息（min / max / mean 等）。

    **kwargs :
        传递给具体统计函数的额外参数。
        例如：
            - 自定义统计所需的额外控制参数

    Returns
    -------
    output : raster-like or ndarray
        若 out_path 为 None：
            返回计算结果数组（或 rastertool 的内存对象）
        若 out_path 提供：
            返回写出后的输出对象

    Notes
    -----
    1. 所有统计均仅基于邻域内的有效像元（NoData 不参与计算）
    2. 内部使用 cache 机制，避免重复计算邻域 count、sum 等中间量
    3. 方差使用总体方差定义（除以 N，而非 N-1）

       
    """
    
    
    
    # ---------- kernel ----------
    if Round:
        kernel = Round_kernel(radius * 2 + 1, deleted)
    else:
        kernel = Rectangle_kernel(radius * 2 + 1, deleted)

    # ---------- read ----------
    data, profile = read(source, masked=True)
    valid_mask = (~data.mask).astype('uint8')
    values = data.filled(0)

    # ---------- cache ----------
    cache = {}
    # cache["count"] = convolve(valid_mask, kernel, mode=mode, cval=cval)

    # ---------- dispatch ----------
    if stat in FOCAL_STATS:
        func = FOCAL_STATS[stat]
    elif callable(stat):
        func = stat
    else:
        raise ValueError(
            f"Unknown stat '{stat}', must be one of {list(FOCAL_STATS.keys())} or callable"
        )

    result = func(
        values,
        valid_mask,
        kernel,
        mode=mode,
        cval=cval,
        cache=cache,
        q=q,
        **kwargs,
    )

    # ---------- mask ----------
    if crop:
        mask = data.mask
    else:
        mask = focal_count(values, valid_mask, kernel, mode=mode, cval=cval, cache=cache) == 0

    # ---------- output ----------
    return output(
        out_path,
        result,
        mask,
        profile,
        nodata,
        dtype,
        compress,
        update_stats,
    )


def focaltool_array(
    source,
    radius,
    stat,
    *,
    out_path=None,
    deleted=False,
    q=50,
    Round=True,
    mode="reflect",
    cval=0.0,
    crop=True,
    **kwargs,
):
    
    # ---------- kernel ----------
    if Round:
        kernel = Round_kernel(radius * 2 + 1, deleted)
    else:
        kernel = Rectangle_kernel(radius * 2 + 1, deleted)

    # ---------- read ----------
    # data, profile = read(source, masked=True)
    valid_mask = (~source.mask).astype('uint8')
    values = source.filled(0)

    # ---------- cache ----------
    cache = {}
    # cache["count"] = convolve(valid_mask, kernel, mode=mode, cval=cval)

    # ---------- dispatch ----------
    if stat in FOCAL_STATS:
        func = FOCAL_STATS[stat]
    elif callable(stat):
        func = stat
    else:
        raise ValueError(
            f"Unknown stat '{stat}', must be one of {list(FOCAL_STATS.keys())} or callable"
        )

    result = func(
        values,
        valid_mask,
        kernel,
        mode=mode,
        cval=cval,
        cache=cache,
        q=q,
        **kwargs,
    )

    # ---------- mask ----------
    if crop:
        mask = source.mask
    else:
        mask = focal_count(values, valid_mask, kernel, mode=mode, cval=cval, cache=cache) == 0

    # ---------- output ----------
    return result, mask














