# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:17:29 2026

@author: wly
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rastertool.functions import get_dataset_opener



def dummy_raster(source, dest_dir, us=None, values_dict=None):
    """
    根据输入分类栅格生成多个二值栅格。

    该函数读取一个分类栅格数据，将每个唯一值分别提取为一个
    单独的二值栅格文件。输出栅格中：
    
    - 目标类别像元值为 1
    - NoData 区域值为 255
    - 其他区域为 0

    每个类别输出一个 GeoTIFF 文件，文件名由 `values_dict`
    指定的名称或像元值决定。

    Parameters
    ----------
    source : str
        输入栅格文件路径（GeoTIFF 或其他 rasterio 支持格式）。
        通常为一个分类栅格，其中不同整数值代表不同类别。

    dest_dir : str
        输出目录路径。生成的二值栅格文件将保存到该目录。

    values_dict : dict, optional
        分类值与输出文件名的映射字典，格式为：
        
        {像元值: 名称}
        
        例如：
        
        {1: '残积物', 2: '冲积物', 3: '红黏土'}
        
        如果未提供或字典中不存在某个值，则使用像元值字符串
        作为输出文件名。

    Returns
    -------
    None
        函数无返回值。结果以 GeoTIFF 文件形式写入 `dest_dir`。

    Notes
    -----
    输出栅格数据类型为 uint8：
    
    - 1 表示目标类别
    - 0 表示非目标类别
    - 255 表示 NoData
    
    每个唯一类别值都会生成一个单独的栅格文件。

    Examples
    --------
    >>> values_dict = {
    ...     1: '残积物',
    ...     2: '冲积物',
    ...     3: '红黏土'
    ... }
    >>> dummy_raster('soil.tif', 'out_dir', values_dict)
    """

    dataset_opener = get_dataset_opener(source)

    if values_dict is None:
        values_dict = {}

    with dataset_opener(source) as src:

        # 读取栅格数据
        data = src.read(masked=True)
        data_mask = data.mask

        shape = data.shape
        profile = src.profile

        # 设置输出栅格参数
        nodata = 255
        dtype = 'uint8'
        profile.update(nodata=nodata, dtype=dtype)

        # 获取唯一有效值
        if us is None:
            us = np.unique(data[~data_mask])

        # 遍历每个类别
        for u in us:

            # 创建输出数组
            dest = np.zeros(shape, dtype=dtype)

            # 获取类别名称
            name = values_dict.get(u, str(u))

            # 赋值
            np.copyto(dest, 1, where=(data == u))
            np.copyto(dest, nodata, where=data_mask)

            # 输出
            out_ph = os.path.join(dest_dir, name + '.tif')

            with rasterio.open(out_ph, 'w', **profile) as dst:
                dst.write(dest)
        



if __name__ == '__main__':
    ph_tif = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/test/dummy_raster/data/抚州市母质母质.tif'


    out_dir = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/test/dummy_raster/data/out1'
    
    
    values = []
    names = []
    values_dict = {1: '残积物',
                   2: '冲积物',
                   3: '红黏土',
                   4: '洪冲积',
                   5: '坡积物',
                   6: '其他'}
    
    dummy_raster(ph_tif, out_dir, values_dict=values_dict)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

























