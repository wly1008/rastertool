# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 19:05:11 2026

@author: wly
"""

from rastertool.core.merge import merge_distance_weight,MERGE_METHODS
import os



from glob import glob
os.chdir(r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/tast')

dir_tif = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/tast/data/OM/source data'
out_dir = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/tast/data/OM/merge'

tif_files = glob(os.path.join(dir_tif, '*.tif'))   # 相同空间参考




buffer = 1000
sampling = 30

W = buffer * 2 / sampling   # 过渡带宽度
sigma =  W / 3   # W ≈ 3σ




MERGE_METHODS.keys()

methods = ['Gaussian', 'IDW', 'Voronoi']
method_kwds_dict = {'Gaussian': {'W':W, 'sigma': sigma},
               'IDW': {'W': W, 'k': 2},
               'Voronoi': None,
               }

## 默认参数
for method in methods:
    
    method_kwds = None
    
    
    out_ph = os.path.join(out_dir, '抚州_%s.tif' % method)
    
    merge_distance_weight(tif_files,
                          method=method,
                          method_kwds=method_kwds,
                          dtype='float32',
                          dst_path=out_ph,
                          target_aligned_pixels=0,
                          bar_name=method)

# ## 自定义参数
# out_dir = r'D:/app/anaconda3/envs/py312/Lib/rastertool/tast/data/OM/merge/custom'
# for method in methods[:2]:
    
#     # method_kwds = None
#     method_kwds = method_kwds_dict[method]
    
    
#     out_ph = os.path.join(out_dir, '抚州_%s.tif' % method)
    
#     merge_distance_weight(tif_files,
#                           method=method,
#                           method_kwds=method_kwds,
#                           dtype='float32',
#                           dst_path=out_ph,
#                           target_aligned_pixels=0,
#                           bar_name=method)


















