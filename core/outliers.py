# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 10:21:34 2026

@author: wly
"""

import numpy as np
from functools import partial


from rastertool.functions import readarray, set_nodata, get_dataset_opener, out


def threshold_from_values(value_range, *_):
    if value_range is None:
        raise ValueError('请设置有效值范围')
    
    q_low, q_high = value_range
    return q_low, q_high





def threshold_from_std(std_multiple, values):
    if std_multiple is None:
        std_multiple = 3
    
    mean = np.mean(values)
    std = np.std(values)
    
    q_low  = mean - std_multiple * std
    q_high = mean + std_multiple * std
    return q_low, q_high
    


def threshold_from_perc(percentage, values):
    if percentage is None:
        percentage = 5
    

    try:
        percs = sorted([float(percentage) , 100 - float(percentage)])
    except TypeError:
        percs = (
            sorted([float(percentage) , 100 - float(percentage)])
            if len(percentage) == 1
            else percentage[0:2]
        )

    q_low, q_high = np.percentile(values, percs)
    
    return q_low, q_high
    

    
def copyto_threshold(dest, src_data, q_low, q_high, nodata ,valid_mask):
    
    
    # 剔除异常值
    outlier_mask = valid_mask & (
                                (src_data < q_low) | (src_data > q_high)
                                )

    np.copyto(dest, nodata, where=outlier_mask)



def copyto_valid(dest, src_data ,src_mask, area_mask=None):
    # 提取区域值
    if area_mask is None:
        area_mask = np.full(src_data.shape, True, dtype=bool)
    
    
    valid_mask = src_mask & area_mask
    
    values = src_data[valid_mask]
    if np.isnan(values).any():
        raise ValueError('有效值位置存在nan')
    
    np.copyto(dest, src_data, where=valid_mask)
    
    return values, valid_mask



def copyto_method(dest, src_data, nodata ,threshold, src_mask, area_mask=None):
    '''mask为有效值位置掩膜'''
    
    values, valid_mask = copyto_valid(dest, src_data ,src_mask, area_mask)
    
    q_low, q_high = threshold( values)
    
    copyto_threshold(dest, src_data, q_low, q_high, nodata ,valid_mask)
    
    



MERGE_METHODS = {
    "perc": threshold_from_perc,
    "std": threshold_from_std,
    "value": threshold_from_values,

}



def remove_outliers(source, dst_in=None, out_path=None,
                method='std', method_arg=None, drop=False,
                nodata=None, dtype=None, compress=None, update_stats=False):
    
    """
    对栅格数据进行异常值（outliers）剔除处理。

    根据指定的方法计算阈值范围，将超出阈值的像元置为 nodata。
    支持整幅栅格统一处理，或按区域栅格（dst_in）进行分区独立剔除。

    Parameters
    ----------
    source : str or rasterio.DatasetReader
        输入栅格数据路径，或可被 rasterio 打开的数据源。

    dst_in : str or array-like, optional
        区域分区栅格或数组。
        若提供，则对每个区域（唯一值）分别计算阈值并进行异常值剔除，
        各区域之间互不影响。
        None 表示对整幅栅格统一处理（默认）。

    out_path : str, optional
        输出栅格文件路径。
        若为 None，则不写文件，直接返回结果数组和 profile（默认）。

    method : {'std', 'perc', 'value'} or callable, optional
        异常值判定方法（默认 'std'）：
        - 'std'   : 基于均值 ± N 倍标准差
        - 'perc'  : 基于百分位阈值
        - 'value' : 直接指定有效值范围
        - callable : 自定义阈值函数，需返回 (q_low, q_high)

    method_arg : int, float, tuple or sequence, optional
        阈值方法的参数：
        - method='std'   : 标准差倍数（如 3）
        - method='perc'  : 百分位数（如 5 或 (5, 95)）
        - method='value' : 有效值范围 (min, max)
        若为 None，则使用各方法的默认参数，如 method='value'，则为必填项。

    drop : bool, optional
        是否丢弃未参与分区处理的像元（默认 False）。
        当 dst_in 不为 None 时：
        - False：未参与分区的像元保留原值
        - True ：未参与分区的像元保持为 nodata

    nodata : number, optional
        输出栅格的 nodata 值。
        若为 None，则继承输入栅格的 nodata 设置。

    dtype : numpy.dtype, optional
        输出栅格的数据类型。
        若为 None，则继承输入栅格的数据类型。

    compress : str, optional
        输出栅格的压缩方式（如 'lzw'）。
        若为 None，则继承输入栅格的压缩设置。

    update_stats : bool, optional
        写出文件时是否更新栅格统计信息（默认 False）。

    Returns
    -------
    dest : numpy.ndarray
        处理后的栅格数组（仅在 out_path 为 None 时返回）。

    profile : dict
        输出栅格的 profile 信息（仅在 out_path 为 None 时返回）。

    Notes
    -----
    - 仅对输入栅格中的有效像元参与阈值计算。
    - nodata 像元始终保持为 nodata，不参与统计与剔除。
    - 分区剔除时，每个区域独立计算阈值。
    - 若提供 out_path，则函数不返回值，结果直接写入文件。
    """
    if method in MERGE_METHODS:
        threshold = MERGE_METHODS[method]
        
    elif callable(method):
        threshold = method
    else:
        raise ValueError(
            "Unknown method {}, must be one of {} or callable".format(
                method, list(MERGE_METHODS.keys())
            )
        )
    threshold = partial(threshold, method_arg)
    
    
    dataset_opener = get_dataset_opener(source)
    
    
    
    
    with dataset_opener(source) as src:
        
        data = src.read(masked=True)
        data_mask = data.mask
        
        dt = src.dtypes[0]
        shape = data.shape
        nodataval = src.nodatavals[0]
        
        nodata, dtype = set_nodata(nodataval, dt)
        
        
        profile = src.profile
        compress = profile.get('compress', None) if compress is None else compress
        profile.update({'nodata': nodataval, 'dtype': dt, 'compress':compress})
        
        dest = np.empty(shape, dtype=dtype)
        
        if dst_in is not None:
            areas = readarray(dst_in, masked=True)
            areas_mask = areas.mask
            uniques = np.unique(areas.compressed())
            
            
            for unique in uniques:
                area_maskn = areas == unique
                
                copyto_method(dest, data, nodata, threshold, ~data_mask, area_maskn)
                
            otr = (~areas_mask) & (~data_mask)
            
            if not drop:
                np.copyto(dest, data, where=otr)
            
            
        else:
            copyto_method(dest, data, nodata, threshold, ~data_mask)
        np.copyto(dest, nodata, where=data_mask)
        
        if out_path is None:
            return dest, profile
        else:
            out(out_path, dest, profile, update_stats=update_stats)



# def fill_outliers(source, dst_in=None, out_path=None,
#                 method='std', method_arg=None, drop=False,
#                 nodata=None, dtype=None, compress=None, update_stats=False):
    
    
#     dest, profile = remove_outliers(source, dst_in=dst_in, out_path=None,
#                                     method=method, method_arg=method_arg, drop=drop,
#                                     nodata=nodata, dtype=dtype,)
    
#     data = readarray(source)
    
    
    
    
    
    
    




































