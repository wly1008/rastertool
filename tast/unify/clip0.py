# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:31:38 2024

@author: wly
"""
from mycode.rio_wrap.core._unify import _unify






def clip(
         raster_in, dst_in=None, out_path=None,
         get_ds=True,
         bounds=None, 
         mode='round',
         nodata = None,
         projection='geographic',
         with_complement=True,
         crop=False,
         arr_crop=None,
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

    delete : TYPE, optional
        是否删除输入栅格raster_in（清除中间变量）,
        当raster_in为地址时正常执行, 而输入栅格类变量时不会, 除非使用'!True'.
        The default is False.
    **creation_options : TYPE
        输出栅格其他profile更新选项


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
    return _unify(
                  raster_in, dst_in=dst_in, out_path=out_path,
                  get_ds=get_ds,
                  bounds=bounds, 
                  mode=mode,
                  nodata = nodata,
                  projection=projection,
                  with_complement=with_complement,
                  crop=crop,
                  arr_crop=arr_crop,
                  unify=0, 
                  delete=delete,
                  stacklevel=3,
                  **creation_options,
                  )
    
    
    
    
    
    
















