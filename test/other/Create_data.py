# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:41:35 2024

@author: wly
"""

import rasterio,os
from rasterio.profiles import default_gtiff_profile
import numpy as np
import pandas as pd
from rasterio.transform import from_origin

os.chdir(r'F:\PyCharm\pythonProject1\代码\mycode\测试文件')


pro = default_gtiff_profile


crs='+proj=latlong'

west, north = 111.3, 38.6
width, height = 2, 2
ras = 1
count=1

transform = from_origin(west, north, ras, ras)

arr = np.arange(1, int(height*width)+1).reshape(width,height)




tif_ph = f'源数据/{west}_{north}_{ras}_5_5.tif'


with rasterio.open(tif_ph, 'w',
                   driver='GTiff',
                   height=height,
                   width=width,
                   nodata=0,
                   count=1,
                   dtype='int16',
                   crs='+proj=latlong',
                   transform=transform,) as dst:
    dst.write(arr,1)
    



















































