# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:02:59 2026

@author: wly
"""

import numpy as np
from osgeo import gdal
def hex_to_rgba(hex_color, A=255):
    """
    将十六进制颜色值转换为 RGBA 元组
    :param hex_color: 十六进制颜色字符串，支持带 # 或不带 #，如 "#FF0000"、"FF0000"、"F00"
    :param A: 透明度值，范围 0-255，默认 255（不透明）
    :return: (r, g, b, a) 元组，每个值范围 0-255
    """
    # 移除开头的 # 号，并转为大写（增强兼容性）
    hex_color = hex_color.lstrip('#').upper()
    
    # 处理短格式（如 #F00 转为 #FF0000）
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    
    # 校验长度，避免索引越界
    if len(hex_color) != 6:
        raise ValueError("无效的十六进制颜色格式，请输入 6 位（或 3 位）十六进制字符串，如 #FF0000")
    
    # 转换为 RGB 值
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # 确保透明度在合法范围
    A = max(0, min(255, A))
    
    return (r, g, b, A)

def colormap(source, class_colors):
    
    
    ds = gdal.Open(source, gdal.GA_Update)
    if ds is None:
       raise FileNotFoundError(f"无法打开文件: {source}")
    band = ds.GetRasterBand(1)
    if band is None:
            raise RuntimeError("无法获取第一个波段")
    
    
    color_table = gdal.ColorTable()

    for value, color in class_colors.items():
        raster_value = int(value)
        # 根据颜色类型处理
        if isinstance(color,str):
            rgba = hex_to_rgba(color)
        else:
            if len(color) == 3:
                rgba = (*color,255)
            else:
                rgba = color
        # 设置颜色表条目
        color_table.SetColorEntry(raster_value, rgba)

    band.SetRasterColorTable(color_table)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    # 刷新数据确保写入
    ds.FlushCache()
    ds = None

def build_raster_attribute_table(raster_path, class_names=None, class_colors=None):

    ds = gdal.Open(raster_path, gdal.GA_Update)
    band = ds.GetRasterBand(1)

    data = band.ReadAsArray()

    values, counts = np.unique(data, return_counts=True)

    # 创建 Raster Attribute Table
    rat = gdal.RasterAttributeTable()

    rat.CreateColumn("VALUE", gdal.GFT_Integer, gdal.GFU_MinMax)
    rat.CreateColumn("COUNT", gdal.GFT_Integer, gdal.GFU_PixelCount)

    if class_names:
        rat.CreateColumn("CLASS_NAME", gdal.GFT_String, gdal.GFU_Name)

    rat.SetRowCount(len(values))

    for i, (v, c) in enumerate(zip(values, counts)):

        rat.SetValueAsInt(i, 0, int(v))
        rat.SetValueAsInt(i, 1, int(c))

        if class_names:
            rat.SetValueAsString(i, 2, class_names.get(v, ""))

    band.SetDefaultRAT(rat)

    # ----------------------
    # 创建颜色表
    # ----------------------

    if class_colors:

        color_table = gdal.ColorTable()

        for value, color in class_colors.items():
            r, g, b = color
            color_table.SetColorEntry(int(value), (r, g, b, 255))

        band.SetRasterColorTable(color_table)
        band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    ds = None