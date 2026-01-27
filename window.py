# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 19:31:33 2026

@author: wly
"""

import warnings
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
from rastertool.functions import get_dataset_opener






def window(raster_in,
           shape=None, size=None,
           step=None,
           get_dict_id_win=False , get_dict_id_self_win=False,
           get_self_wins=False,
           initial_offset=None,
           Tqbm=False):
    '''
    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        栅格数据或栅格地址
    shape : tuple
          (height,width)
        分割为 height*width个窗口, 未除尽的并入末端窗口
    size : int、float or tuple
          (ysize,xsize)
        窗口的尺寸大小，多余的会生成独立的小窗口不会并入前一个窗口
        
    step : tuple or int
          (ystep,xstep)
        生成滑动窗口
        为滑动步进
        shape、size参数都可以与之配合使用，这里的shape代表了窗口的尺寸为总长、宽除以shape的向下取整。
        e.g.
        src.shape = (20,20)
        shape:(3,3) == size:(6,6)
        末端窗口按正常步进滑动，如有超出会剔除多余部分
        如填int类型，ystep = xstep = step;
        如tuple中存在None,则相应的维度取消滑动，或者说滑动步进等于窗口尺寸。
        e.g.
        3 -> (3,3)
        (3,None) -> (3,xsize)
        (None,3) -> (ysize,3)
    get_self_wins : bool
        如使用滑动窗口是否返回去覆盖后的自身窗口
        
    initial_offset : tuple
                    (initial_offset_x, initial_offset_y)
        初始偏移量,默认为(0,0)

    Returns
    -------
    windows : TYPE
        窗口集
    inxs : TYPE
        对应窗口在栅格中的位置索引

    '''

    assert shape or size, '请填入shape or size'
    assert not (shape and size), 'shape 与 size只填其中一个'

    dataset_opener = get_dataset_opener(raster_in)
    
    with dataset_opener(raster_in) as src:
        
        if size:
            if isinstance(size, (int ,float)):
                xsize = size
                ysize = size
            else:
                ysize, xsize = size
            xend = src.width % xsize or xsize
            yend = src.height % ysize or ysize
            
            
            s0 = int(np.ceil(src.height / ysize))
            s1 = int(np.ceil(src.width / xsize))
            shape = (s0, s1)
            
        else:
            xsize, xend0 = divmod(src.width, shape[1])
            ysize, yend0 = divmod(src.height, shape[0])
            xend = xsize + xend0
            yend = ysize + yend0
        
        
        # 生成滑动窗口
        if step:
            
            # 获取x、y 步进
            if isinstance(step, int):
                xstep = step
                ystep = step
            else:
                xstep = step[1] or xsize
                ystep = step[0] or ysize
            # 步进过大，存在缝隙
            if xstep > xsize:
                warnings.warn('步进大于窗口尺寸：xstep > xsize')
            if ystep > ysize:
                warnings.warn('步进大于窗口尺寸：ystep > ysize')
            
            # 计算窗口数
            s00, yend0 = divmod(src.height - ysize, ystep)
            s10, xend0 = divmod(src.width - xsize, xstep)
            
            s0 = int(s00+1 if yend0 == 0 else s00+2)
            s1 = int(s10+1 if yend0 == 0 else s10+2)
            shape = (s0, s1)
            
            # 末端窗口修减
            yend = ysize - (ystep - (yend0 or ystep))
            xend = xsize - (xstep - (xend0 or xstep))
    
        else:
            # 规范变量
            xstep = None
            ystep = None
        
        initial_offset_x, initial_offset_y = initial_offset or (0,0)  # 初始偏移量
        
        # 返回值变量
        inxs = []  # 窗口位置索引
    
        windows = []
        if get_self_wins:
            self_windows = []
        
    
        if Tqbm:
            pbar = tqdm(total=shape[0]*shape[1], desc='生成窗口')
    
        y_off = initial_offset_y  # y初始坐标
        for y_inx,ax0 in enumerate(range(shape[0])):
            
            x_off = initial_offset_x
            height = yend if ax0 == (shape[0] - 1) else ysize 
            if height == 0:
                continue
            if get_self_wins:
                self_height = yend if ax0 == (shape[0] - 1) else ystep
                
            for x_inx,ax1 in enumerate(range(shape[1])):
    
                width = xend if ax1 == (shape[1] - 1) else xsize
                if width == 0:
                    continue
                if get_self_wins:
                    self_width = xend if ax1 == (shape[1] - 1) else xstep
                
                windows.append(Window(x_off, y_off, width, height))
                if get_self_wins:
                    self_windows.append(Window(x_off, y_off, self_width, self_height))
                
    
                inxs.append((y_inx,x_inx))
                
                x_off += xstep or width
                if Tqbm:
                    pbar.update(1)
    
            
            y_off += ystep or height
        if Tqbm:
            pbar.close()
    
        return (windows, inxs) if not get_self_wins else (windows, inxs, self_windows)







def get_attribute_from_window(window):
    '''
    获取窗口属性

    Parameters
    ----------
    window : TYPE
        DESCRIPTION.

    Returns
    -------
    col_off : TYPE
        DESCRIPTION.
    row_off : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    height : TYPE
        DESCRIPTION.

    '''
    
    if isinstance(window, Window):
        col_off, row_off, width, height = window.col_off, window.row_off, window.width, window.height
    else:
        col_off, row_off, width, height = window
    return col_off, row_off, width, height

def expand_window(window, size, mod='around', trim=True, Rwidth=None, Rheight=None):
    '''
    扩张窗口边界，mod为四周型

    Parameters
    ----------
    window : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.
    mod : TYPE, optional
        DESCRIPTION. The default is 'around'.
    trim : TYPE, optional
        DESCRIPTION. The default is True.
    Rwidth : TYPE, optional
        DESCRIPTION. The default is None.
    Rheight : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    win : TYPE
        DESCRIPTION.

    '''
    
    # 参数条件检验
    if trim:
        if Rwidth is None or Rheight is None:
            raise ValueError("trim为True ，检测到Rwidth 和 Rheight 存在None，请传入有效值！")
    
    # 获取窗口位置参数
    if isinstance(window, Window):
        col_off, row_off, width, height = window.col_off, window.row_off, window.width, window.height
    else:
        col_off, row_off, width, height = window
    
    # 选择扩张模式
    if mod == 'around':
        col_off -= size
        row_off -= size
        width += 2*size
        height += 2*size
        win = Window(col_off, row_off, width, height)
    
    # 是否修剪
    if trim:
        win = trim_window(win, Rwidth, Rheight)
    
    return win
    

def trim_window(window, Rwidth, Rheight):
    '''
    修剪超出源栅格的范围
    eg. (Rwidth, Rheight) == (100,100); (-1,2,10,10)-->(0,2,9,10)

    Parameters
    ----------
    window : TYPE
        DESCRIPTION.
    Rwidth : TYPE
        DESCRIPTION.
    Rheight : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    
    # 获取位置参数
    if isinstance(window, Window):
        col_off, row_off, width, height = window.col_off, window.row_off, window.width, window.height
    else:
        col_off, row_off, width, height = window
    
    right = col_off + width
    Bottom = row_off + height
    
    
    # 判断是否超出范围，并修剪
    col_off = 0 if col_off < 0 else col_off
    row_off = 0 if row_off < 0 else row_off
    
    # 如果窗口右或下边界不超出总边界则用自身边界计算，否则用总边界作为新的边界
    width = right - col_off if right < Rwidth else Rwidth - col_off
    height = Bottom - row_off if Bottom < Rheight else Rheight - row_off
    
    return Window(col_off, row_off, width, height)




def is_window_contains(window, x_window):
    """
    判断window是否完全包含x_window
    :param window: 被判断的大窗口，格式 (col_off, row_off, width, height)
    :param x_window: 被包含的小窗口，格式 (col_off, row_off, width, height)
    :return: 包含返回True，不包含返回False
    """
    # 解包两个窗口的参数
    w_col, w_row, w_w, w_h = get_attribute_from_window(window)
    x_col, x_row, x_w, x_h = get_attribute_from_window(x_window)
    
    # 计算两个窗口的右边界、下边界
    w_col_end = w_col + w_w
    w_row_end = w_row + w_h
    x_col_end = x_col + x_w
    x_row_end = x_row + x_h
    
    # 判断4个核心条件，全部满足则包含
    cond1 = w_col <= x_col          # 左边界：大窗在左
    cond2 = w_row <= x_row          # 上边界：大窗在上
    cond3 = w_col_end >= x_col_end  # 右边界：大窗在右
    cond4 = w_row_end >= x_row_end  # 下边界：大窗在下
    return all([cond1, cond2, cond3, cond4])

def get_array_from_window(x_window, window, array):
    '''
    只支持window包含x_window的情况
    
    
    '''
    if not is_window_contains(window, x_window):
        raise ValueError("只支持window包含x_window的情况")
    
    # 计算相对偏移
    rel_row = x_window.row_off - window.row_off
    rel_col = x_window.col_off - window.col_off
    
    # 截取x_array
    x_array = array[rel_row:rel_row+x_window.height, rel_col:rel_col+x_window.width]
    
    return x_array






























