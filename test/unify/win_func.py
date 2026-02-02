# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:57:38 2023

@author: wly
"""

import mycode.arcmap as ap
from mycode.arcmap import window,get_RasterAttr


import rasterio
import rasterio.mask
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
from rasterio.warp import reproject as _reproject



import pathlib,re
from sys import getsizeof
import numpy as np



def wins_to_rasters(raster_in,windows):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass






def reproject(raster_in, dst_in=None,
              out_path=None, get_ds=True,
              maxsize=1024*3,
              crs=None,
              how='mode',
              run_how=None,
              resolution=None, shape=(None, None, None)):
    
    
    src = rasterio.open(raster_in) if issubclass(type(raster_in), (str,pathlib.PurePath)) else raster_in
    if crs:
        pass
    elif dst_in:
        crs = get_RasterAttr(dst_in, 'crs')
    else:
        raise Exception("dst_in 和 bounds必须输入其中一个")
    
    
    # dtypes = src.dtypes
    
    # try:
    #     sizes = [int(re.findall(r'\d+$',s)) if isinstance(s, str) else int(re.findall(r'\d+$',s.__name__)) for s in dtypes]
    # except:
        
        
    h,w = src.height,src.width
    one = src.read(window=Window(0,0,1,1))
    
    sizes = getsizeof(one)-getsizeof(np.array([[[]]]))
    
    d = int((maxsize / sizes)**0.5)
    
    
    how = how if isinstance(how, int) else getattr(Resampling, how)
    windows,ids = window(src,size=d)
    
    for win in windows:
        boundn = src.window_bounds(win)
        widthn = win.width
        heightn = win.height
        dst_transform, dst_width, dst_height = calculate_default_transform(src.crs, crs, widthn, heightn, *boundn)
        
        
    
    
    pass


x = Window(0,0,500,500)


x.crop(1000, 1000)[2]







arr = np.array([[1,2],[1,2]])

raster_in = r'F:/PyCharm/pythonProject1/arcmap/015温度/土地利用/landuse_4y/1990-5km-tiff.tif'
dst_in = r'F:\PyCharm\pythonProject1\arcmap\007那曲市\data\eva平均\eva_2.tif'
reproject(raster_in,dst_in)

a = np.arange(10000*10000)


from sys import getsizeof as getsize
var = object()
print(getsize(a))




def binary_conversion(var: int):
    """
    二进制单位转换
    :param var: 需要计算的变量，bytes值
    :return: 单位转换后的变量，kb 或 mb
    """
    assert isinstance(var, int)
    if var <= 1024:
        return f'占用 {round(var / 1024, 2)} KB内存'
    else:
        return f'占用 {round(var / (1024 ** 2), 2)} MB内存'









a = np.array([1]).astype('int32')

for i in a:
    getsizeof(i)

# sum([getsizeof(i) for i in a])















