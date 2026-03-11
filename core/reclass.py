# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:58:01 2026

@author: wly
"""

import os, re,cmath
import numpy as np
import pandas as pd
import rasterio

from rastertool.functions import  get_dataset_opener

def hex_to_rgba(hex_color, A=255):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, A)


def build_mask(data, expr):
    
    expr = str(expr).strip()
    
    # 单个数值
    if re.fullmatch(r"-?\d+(\.\d+)?", expr):
        v = float(expr)
        return data == v
    
    # 区间 10-30
    if re.fullmatch(r"-?\d+(\.\d+)?\s*-\s*-?\d+(\.\d+)?", expr):
        a, b = re.split(r"\s*-\s*", expr)
        a = float(a)
        b = float(b)
        return (data >= a) & (data <= b)
    
    # >= <= > <
    m = re.fullmatch(r"(>=|<=|>|<)\s*(-?\d+(\.\d+)?)", expr)
    if m:
        op = m.group(1)
        v = float(m.group(2))
        
        if op == ">=":
            return data >= v
        elif op == "<=":
            return data <= v
        elif op == ">":
            return data > v
        elif op == "<":
            return data < v
    
    raise ValueError(f"Unsupported expression: {expr}")




    
    
def reclass(source, dest_path, new_values, old_values, 
            color_list=None, color_type=None,
            dtype=None, nodata=None,
            other_to_nodata=False):
    """
    栅格重分类函数

    参数：
    new_values : list
        新分类值列表
    old_values : list
        原始值（或条件）列表，与 new_values 一一对应
    dest_path : str
        输出栅格路径
    color_list : list, optional
        每个分类对应的颜色
    color_type : str, optional
        颜色类型，可选 'hex' 或 'rgba'
    dtype : numpy dtype, optional
        输出栅格数据类型
    nodata : number, optional
        输出栅格无效值
    other_to_nodata : bool
        未被匹配的值是否设为 nodata
    """

    # 分类数量
    number = len(new_values)

    # 确保 old_values 与 new_values 数量一致
    assert len(old_values) == number

    # =========================
    # 构建 colormap（颜色表）
    # =========================
    if color_list is not None:

        # 颜色数量必须和分类数量一致
        assert len(color_list) == number

        # colormap 只支持 uint8 类型栅格
        assert np.dtype(dtype) == np.uint8, 'colormap仅支持unit8格式栅格'

        # hex 格式颜色 (#RRGGBB)
        if color_type == 'hex':
            colormap = {
                v: hex_to_rgba(c)  # hex 转 RGBA
                for v, c in zip(new_values, color_list)
            }

        # rgba 元组格式 (R,G,B,A)
        elif color_type == 'rgba':
            colormap = {
                v: c
                for v, c in zip(new_values, color_list)
            }

        # 不支持的颜色类型
        else:
            raise Exception('color_type 仅支持 hex 字符串与 rgba 元组格式')

    else:
        # 如果没有提供颜色
        colormap = None

    # =========================
    # 打开数据集
    # =========================
    dataset_opener = get_dataset_opener(source)

    with dataset_opener(source) as src:

        # 读取栅格数据（带 mask）
        data = src.read(masked=True)

        # 数据的 mask（True 表示 nodata）
        data_mask = data.mask

        # 栅格形状 (band, height, width)
        shape = data.shape

        # 原始 profile 信息
        profile = src.profile

        nodataval = profile['nodata']
        dt = profile['dtype']

        # =========================
        # dtype 与 nodata 处理
        # =========================

        if nodata is None:
            nodata = nodataval

        if dtype is None:
            dtype = dt

        # numpy dtype 对象
        src_dtype = np.dtype(dt)
        dst_dtype = np.dtype(dtype)

        # =========================
        # nodata 与 dtype 合法性检查
        # =========================

        inrange = False

        # 如果是整数类型
        if np.issubdtype(dtype, np.integer):

            # 获取整数类型范围
            info = np.iinfo(dtype)

            # 检查 nodata 是否在范围内
            inrange = info.min <= nodata <= info.max

        else:
            # 浮点类型

            if cmath.isfinite(nodata):

                info = np.finfo(dt)

                # 检查 nodata 是否在浮点范围
                inrange = info.min <= nodata <= info.max

                # 检查是否可安全转换
                nodata_dt = np.min_scalar_type(nodata)

                inrange = inrange & np.can_cast(nodata_dt, dt)

            else:
                # nodata 为 inf 或 nan
                inrange = True

        # 如果 nodata 与 dtype 不匹配
        if not inrange:
            raise Exception('nodata 与 dtype 不匹配')

        # =========================
        # 初始化输出数组
        # =========================

        # 输出栅格
        dest = np.zeros(shape, dtype=dtype)

        # 已经被赋值的位置
        assigned = np.zeros(shape, dtype=bool)

        # =========================
        # 重分类主循环
        # =========================
        for i in range(number):

            # 新值
            new_value = new_values[i]

            # 旧值（或条件）
            old_value = old_values[i]

            # 构建匹配 mask
            mask = build_mask(data, old_value)

            # 排除：
            # 1 已经赋值的位置
            # 2 原始 nodata
            mask = mask & (~assigned) & (~data_mask)

            # 将 new_value 复制到目标数组
            np.copyto(dest, new_value, where=mask)

            # 更新 assigned 标记
            assigned |= mask

        # =========================
        # 未匹配像元处理
        # =========================

        # 未匹配值保留原值
        if not other_to_nodata:

            # 未被分类且不是 nodata 的像元
            remain = (~assigned) & (~data_mask)

            if remain.any():

                # 检查原始 dtype 是否能安全转换到目标 dtype
                if not np.can_cast(src_dtype, dst_dtype, casting='safe'):

                    # 如果不能安全转换，则恢复为原始 dtype
                    dtype = dt
                    dest = dest.astype(src_dtype)

                # 保留原始值
                np.copyto(dest, data, where=remain)

                # 原始 nodata 写入
                np.copyto(dest, nodata, where=data_mask)

        else:
            # 未匹配像元全部设为 nodata
            np.copyto(dest, nodata, where=~assigned)

        # =========================
        # 更新 profile
        # =========================
        profile.update({
            'nodata': nodata,
            'dtype': dtype
        })

        # =========================
        # 写入新栅格
        # =========================
        with rasterio.open(dest_path, 'w', **profile) as dst:

            # 写入数据
            dst.write(dest)

            # 更新统计信息
            dst.update_stats()

            # 写入 colormap
            if color_list:
                dst.write_colormap(1, colormap)  

    
    
    
    
    
if __name__ == '__main__':
    
    out_dir = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/test/reclass/data/out'
    ph_table = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/test/reclass/data/table/属性分级.xlsx'

    source = r'D:/app/anaconda3/envs/py313/Lib/site-packages/rastertool/test/reclass/data/阳离子交换量.tif'

    df_tbs= pd.read_excel(ph_table, index_col=0, sheet_name=None)


    df_class = df_tbs['class']
    df_color = df_tbs['color']

    name = '阳离子交换量'
    dfn = df_class[name].dropna()
    new_values = dfn.index.tolist()
    old_values = dfn.values.tolist()


    color_list = df_color[name].dropna().tolist()



    dest_path = os.path.join(out_dir, name + '.tif')


    new_values = new_values; old_values = old_values; dest_path=dest_path;
    color_list=color_list; color_type = 'hex'
    dtype = 'uint8'; nodata=0;
    other_to_nodata=False;
    reclass(new_values, old_values, dest_path, color_list, color_type, dtype, nodata, other_to_nodata)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def reclass(new_values, old_values, dest_path,
    #             color_list=None, color_type=None,
    #             dtype=None, nodata=None,
    #             other_to_nodata=False):
        
    #     number = len(new_values)
    #     assert len(old_values) == number
        
        
    #     if color_list is not None:
            
    #         assert len(color_list) == number
    #         assert np.dtype(dtype) == np.uint8 , 'colormap仅支持unit8格式栅格' 
            
    #         if color_type == 'hex':
    #             colormap = {
    #                 v: hex_to_rgba(c)
    #                 for v, c in zip(new_values, color_list)
    #             }
    #         elif color_type == 'rgba':
    #             colormap = {
    #                 v: c
    #                 for v, c in zip(new_values, color_list)
    #             }
    #         else:
    #             raise Exception('color_type 仅支持 hex 字符串与 rgba 元组格式')
    #     else:
    #         colormap = None
        
        
    #     dataset_opener = get_dataset_opener(source)
        
        
    #     with dataset_opener(source) as src:
            
    #         data = src.read(masked=True)
    #         data_mask = data.mask
    #         shape = data.shape
    #         profile = src.profile
    #         nodataval = profile['nodata']
    #         dt = profile['dtype']
            
            
            
    #         if nodata is None:
    #             nodata = nodataval
    #         if dtype is None:
    #             dtype = dt
    #         src_dtype = np.dtype(dt)
    #         dst_dtype = np.dtype(dtype)
            
            
    #         # 无效值合法判断
    #         inrange = False
    #         if np.issubdtype(dtype, np.integer):
    #             info = np.iinfo(dtype)
    #             inrange = info.min <= nodata <= info.max
    #         else:
    #             if cmath.isfinite(nodata):
    #                 info = np.finfo(dt)
    #                 inrange = info.min <= nodata <= info.max
    #                 nodata_dt = np.min_scalar_type(nodata)
    #                 inrange = inrange & np.can_cast(nodata_dt, dt)
    #             else:
    #                 inrange = True
    #         if not inrange:
    #             raise Exception('nodata 与 dtype 不匹配')
            
        
    #         dest = np.zeros(shape, dtype=dtype)
    #         assigned = np.zeros(shape, dtype=bool)
    #         for i in range(number):
                
    #             new_value = new_values[i]
    #             old_value = old_values[i]
                
                
    #             mask = build_mask(data, old_value)
    #             mask = mask & (~assigned) & (~data_mask)
                
    #             np.copyto(dest, new_value, where=mask)
                
    #             assigned |= mask
            
            
    #         if not other_to_nodata:
    #             remain = (~assigned) & (~data_mask)
                
    #             if remain.any():
    #                 # 检测dt是否可以安全转换为dtype
    #                 if not np.can_cast(src_dtype, dst_dtype, casting='safe'):
    #                     dtype = dt
    #                     dest = dest.astype(src_dtype)
    #                 np.copyto(dest, data, where=remain)
    #                 np.copyto(dest, nodata, where=data_mask)
            
    #         else:
    #             np.copyto(dest, nodata, where=~assigned)
            
    #         profile.update({'nodata': nodata, 'dtype': dtype})
            
    #         with rasterio.open(dest_path, 'w', **profile) as dst:
                
                
    #             dst.write(dest)
    #             dst.update_stats()
    #             if color_list:
    #                 dst.write_colormap(1, colormap)
                
    
    
    


































