# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:11:35 2026

@author: wly
"""

import os
import numpy as np
import pandas as pd
import rasterio
import rastertool



ph_tif = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/tast/data/OM/source data/崇仁县.tif'

out_ph1 = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/tast/data/OM/reproject/崇仁县int8.tif'
out_ph2 = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/tast/data/OM/reproject/崇仁县2.tif'


# rastertool.reproject(ph_tif, crs=4326, out_path=out_ph1, nodata=-1,dtype='int8')



rastertool.reproject(out_ph1, crs=4326, out_path=out_ph2, nodata=-2)







