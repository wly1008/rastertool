# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 10:21:34 2026

@author: wly
"""

import numpy as np
from functools import partial


from rastertool.functions import readarray, set_nodata, get_dataset_opener, out


# def _three_sigma(array,areas=None) -> np.array:
#     '''
#     三倍标准差剔除离散值

#     Parameters
#     ----------
#     array : array_like
#         可正常转为数组的元素
#         需要操作的数组
#     areas : TYPE, optional
#         分区数组列表，每个元素需于数组形状相同，
#         每个元素中的有效值（None、np.nan、False为无效值）为一个区域.
#         （如不同的时间或地区分区剔除，不受其他区域影响）
#         None则不分区，（默认值）
#         The default is None.

#     Returns
#     -------
#     arr : np.array
#         剔除离散值后的数组

#     '''
    
#     arr = np.array(array).astype('float64')
    
#     if areas is None:
#         mean = np.nanmean(arr)
#         std = np.nanstd(arr)
#         arr[(arr < mean - 3 * std) | (arr > mean + 3 * std)] = np.nan
        
#     else:
#         for area in areas:

#             warnings.filterwarnings('ignore',category=RuntimeWarning)
#             area = np.array(area)
#             arrx = np.where((np.isnan(area)|(area==False)|(area==None)),np.nan,arr)
#             mean = np.nanmean(arrx)
#             std = np.nanstd(arrx)
#             arr[(arrx < mean - 3 * std) | (arrx > mean + 3 * std)] = np.nan
#         warnings.filterwarnings('default')

#     return arr



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
    
    
    if method in MERGE_METHODS:
        threshold = MERGE_METHODS[method]
        
    elif callable(method):
        threshold = method
    else:
        raise ValueError(
            "Unknown method {}, must be one of {} or callable".format(
                threshold, list(MERGE_METHODS.keys())
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



def fill_outliers(source, dst_in=None, out_path=None,
                method='std', method_arg=None, drop=False,
                nodata=None, dtype=None, compress=None, update_stats=False):
    
    
    dest, profile = remove_outliers(source, dst_in=dst_in, out_path=None,
                                    method=method, method_arg=method_arg, drop=drop,
                                    nodata=nodata, dtype=dtype,)
    
    data = readarray(source)
    
    
    
    
    
    
    




































