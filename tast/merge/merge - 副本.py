# -*- coding: utf-8 -*-

from contextlib import ExitStack
from scipy.ndimage import distance_transform_edt
import os
import numbers
import math
import cmath
import warnings
import logging
import numpy as np
import pandas as pd
import rasterio

from rastertool.functions import get_dataset_opener, readarray
from tqdm import tqdm

import rasterio
from rasterio.enums import Resampling
from rastertool.errors import (
    MergeError,
    # RasterioDeprecationWarning,
    RastertoolError,
    WindowError,
)

from rasterio.io import DatasetWriter
from rasterio import windows
from rasterio.transform import Affine
from rasterio.windows import subdivide

logger = logging.getLogger(__name__)

def spatial_union(sources):
    dataset_opener = get_dataset_opener(sources[0])
    xs = []
    ys = []

    for i, dataset in enumerate(sources):
        with dataset_opener(dataset) as src:
            src_transform = src.transform

            # if use_highest_res:
            #     best_res = min(
            #         best_res,
            #         src.res,
            #         key=lambda x: x
            #         if isinstance(x, numbers.Number)
            #         else math.sqrt(x[0] ** 2 + x[1] ** 2),
            #     )

            # The merge tool requires non-rotated rasters with origins at their
            # upper left corner. This limitation may be lifted in the future.
            if not src_transform.is_rectilinear:
                raise MergeError(
                    "Rotated, non-rectilinear rasters cannot be merged."
                )
            if src_transform.a < 0:
                raise MergeError(
                    'Rasters with negative pixel width ("flipped" rasters) cannot be merged.'
                )
            if src_transform.e > 0:
                raise MergeError(
                    'Rasters with negative pixel height ("upside down" rasters) cannot be merged.'
                )

            left, bottom, right, top = src.bounds

        xs.extend([left, right])
        ys.extend([bottom, top])

    dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)
    return dst_w, dst_s, dst_e, dst_n


def _intersect_bounds(bounds1, bounds2, transform):
    """Based on gdal_merge.py."""
    int_w = max(bounds1[0], bounds2[0])
    int_e = min(bounds1[2], bounds2[2])

    if int_w >= int_e:
        raise ValueError

    if transform.e < 0:
        # north up
        int_s = max(bounds1[1], bounds2[1])
        int_n = min(bounds1[3], bounds2[3])
        if int_s >= int_n:
            raise ValueError
    else:
        int_s = min(bounds1[1], bounds2[1])
        int_n = max(bounds1[3], bounds2[3])
        if int_n >= int_s:
            raise ValueError

    return int_w, int_s, int_e, int_n

def win_align(window):
    """Equivalent to rounding both offsets and lengths.

    This method computes offsets, width, and height that are
    useful for compositing arrays into larger arrays and
    datasets without seams. It is used by Rasterio's merge
    tool and is based on the logic in gdal_merge.py.

    Returns
    -------
    Window
    """
    row_off = math.floor(window.row_off + 0.1)
    col_off = math.floor(window.col_off + 0.1)
    height = math.floor(window.height + 0.5)
    width = math.floor(window.width + 0.5)
    return windows.Window(col_off, row_off, width, height)

    
def copy_Gaussian(merged_data, new_data,
                  merged_mask, new_mask,
                  dA, dB,
                  W=None,
                  sigma=None,
                  eps=1e-6):

    # 重叠且双方都有值
    validAB = (~merged_mask) & (~new_mask)

    # 先用新数据覆盖旧的 nodata 区域
    np.copyto(merged_data, new_data, where=merged_mask, casting="unsafe")

    if not np.any(validAB):
        return

    # 计算最大距离（仅重叠区）
    if W is None:
        dmax_A = np.max(dA[validAB])
        dmax_B = np.max(dB[validAB])
        W = max(dmax_A, dmax_B)
    else:
        dmax_A = dmax_B = W

    if sigma is None:
        sigma = W / 3.0

    # 初始化权重为 0
    wA = np.zeros_like(dA, dtype=np.float32)
    wB = np.zeros_like(dB, dtype=np.float32)

    # 仅在重叠区计算高斯权重
    xA = (dmax_A - dA[validAB])
    xB = (dmax_B - dB[validAB])

    wA[validAB] = np.exp(-(xA * xA) / (2 * sigma * sigma))
    wB[validAB] = np.exp(-(xB * xB) / (2 * sigma * sigma))

    # 加权融合
    V = (wA * merged_data + wB * new_data) / (wA + wB + eps)

    np.copyto(merged_data, V, where=validAB, casting="unsafe")

def copy_IDW(merged_data, new_data,
             merged_mask, new_mask,
             dA, dB,
             W=None,
             k=2,
             eps=1e-6):

    # 重叠且双方都有值的区域
    validAB = (~merged_mask) & (~new_mask)

    # 先处理“原来为空”的区域：直接用新数据覆盖
    np.copyto(merged_data, new_data, where=merged_mask, casting="unsafe")

    if not np.any(validAB):
        return

    # 计算最大距离（仅基于重叠区）
    if W is None:
        dmax_A = np.max(dA[validAB])
        dmax_B = np.max(dB[validAB])
        W = max(dmax_A, dmax_B)
    else:
        dmax_A = dmax_B = W

    # 初始化权重为 0（非重叠区保持 0）
    wA = np.zeros_like(dA, dtype=np.float32)
    wB = np.zeros_like(dB, dtype=np.float32)

    # 仅在重叠区计算 IDW 权重
    wA[validAB] = 1.0 / ((dmax_A - dA[validAB] + eps) ** k)
    wB[validAB] = 1.0 / ((dmax_B - dB[validAB] + eps) ** k)

    # 加权融合（只在重叠区）
    V = (wA * merged_data + wB * new_data) / (wA + wB + eps)

    np.copyto(merged_data, V, where=validAB, casting="unsafe")


def copy_Average(merged_data, new_data,
             merged_mask, new_mask,
             wA=1, wB=1,
             **kwargs
             ):
    validAB = (~merged_mask) & (~new_mask)
    
    V = (wA*merged_data + wB*new_data) / (wA + wB)
    
    np.copyto(merged_data, new_data, where=merged_mask, casting="unsafe")
    np.copyto(merged_data, V, where=validAB, casting="unsafe")


def copy_Voronoi(merged_data, new_data,
                 merged_mask, new_mask,
                 dA, dB, **kwargs):
    validAB = (~merged_mask) & (~new_mask)

    np.copyto(merged_data, new_data, where=merged_mask)

    if not np.any(validAB):
        return

    choose_new = (dB > dA) & validAB

    np.copyto(merged_data, new_data, where=choose_new, casting="unsafe")


# def copy_Quality(merged_data, new_data,
#                  merged_mask, new_mask,
#                  QA, QB, fill_value=None,**kwargs):
    
#     if fill_value is None:
#         copy_Voronoi(merged_data, new_data,
#                      merged_mask, new_mask,
#                      dA=QA, dB=QB)
#     else:
#         copy_Voronoi(merged_data, new_data,
#                      merged_mask, new_mask,
#                      dA=QA.filled(fill_value), dB=QB.filled(fill_value))
    
#     copy_Voronoi(merged_data=QA, new_data=QB,
#                  merged_mask=QA.mask, new_mask=QB.mask,
#                  dA=QA, dB=QB)

def copy_Quality(merged_data, new_data,
                 merged_mask, new_mask,
                 QA, QB,
                 quality_mode="max",
                 **kwargs):

    validAB = (~merged_mask) & (~new_mask)

    # 原来为空的区域直接覆盖
    np.copyto(merged_data, new_data, where=merged_mask)
    np.copyto(QA, QB, where=merged_mask)

    if not np.any(validAB):
        return

    if quality_mode == "max":
        choose_new = (QB > QA) & validAB
    elif quality_mode == "min":
        choose_new = (QB < QA) & validAB
    else:
        raise ValueError("quality_mode must be 'max' or 'min'")

    np.copyto(merged_data, new_data, where=choose_new)
    np.copyto(QA, QB, where=choose_new)



MERGE_METHODS = {
    "Gaussian": copy_Gaussian,
    "IDW": copy_IDW,
    "Voronoi": copy_Voronoi,
    "Average": copy_Average,
    "Quality": copy_Quality,
    # "max": copy_max,
    # "sum": copy_sum,
    # "count": copy_count,
}


def merge_weight(sources,          # 输入栅格文件列表
                 method='Gaussian', # 加权方法
                 method_kwds=None,  # 方法参数
                 index=1,           # 处理的波段序号
                 bounds=None,       # 输出范围
                 res=None,          # 输出分辨率
                 nodata=None,       # 空值
                 dtype=None,        # 数据类型
                 resampling=Resampling.nearest, # 重采样方法
                 target_aligned_pixels=True,    # 像素对齐
                 masked=False,      # 是否返回掩码数组
                 dst_path=None,     # 输出路径
                 dst_kwds=None,     # 输出参数
                 bar=False):         # 是否显示进度条
    '''
    主体基于rasterio.merge.merge实现，修改重合区域取值算法，实现重合区域加权平均。
    满足带缓冲区的区块拼接的平滑需要

    Parameters
    ----------
    sources : list
        A sequence of dataset objects opened in 'r' mode or Path-like
        objects.
    method : str or callable
        pre-defined method:
            Gaussian: Gaussian distance attenuation function
            IDW: Inverse distance weighting
            average: pixel-wise weighted average of existing and new
        
        or custom callable with signature:
            merged_data : array_like
                array to update with new_data
            new_data : array_like
                data to merge
                same shape as merged_data
            merged_mask, new_mask : array_like
                boolean masks where merged/new data pixels are invalid
                same shape as merged_data
            dA : The shortest Euclidean distance of the pixel distance boundary of merged_data
            dB : The shortest Euclidean distance of the pixel distance boundary of new_data
            **method_kwds : Specific parameters required for weight calculation

        The default is 'Gaussian'.
    method_kwds : dict, optional
        Specific parameters required for weight calculation
        pre-defined method parameters:
            Gaussian:
                W : number, optional. Default: None.
                    叠置区以像元数为单位的宽度.
                    当W为None时，计算dmax,W = dmax.否则以dmax=W.
                sigma : number, optional.  Default: None.
                    标准差，控制衰减速度（越小衰减越快）.当sigma为None时，sigma = W / 3
                    
                eg. 缓冲区w=1000m, 分辨率res=30
                    W = 2w / res;
                    sigma = W / 3
            IDW:
                W : number, optional. Default: None.
                    叠置区以像元数为单位的宽度.
                    当W为None时，计算dmax,W = dmax.否则以dmax=W.
                k : number, optional. Default: 2.
                    距离衰减参数，控制权重随距离衰减的速度。
            Average:
                wA, wB : number,optional. Default: 1.
                对应 merged_data 与 new_data 权重，默认都为1，即简单平均。
        
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).
        If not set, bounds are determined from bounds of input rasters.
    res: tuple, optional
        Output resolution in units of coordinate reference system. If
        not set, a source resolution will be used. If a single value is
        passed, output pixels will be square.
    nodata: float, optional
        nodata value to use in output file. If not set, uses the nodata
        value in the first input raster.
    dtype: numpy.dtype or string
        dtype to use in outputfile. If not set, uses the dtype value in
        the first input raster.
    resampling : Resampling, optional
        Resampling algorithm used when reading input files.
        Default: `Resampling.nearest`.
    target_aligned_pixels : bool, optional
        Whether to adjust output image bounds so that pixel coordinates
        are integer multiples of pixel size, matching the ``-tap``
        options of GDAL utilities.  Default: False.
    masked: bool, optional. Default: False.
        If True, return a masked array. Note: nodata is always set in
        the case of file output.
    dst_path : str or PathLike, optional
        Path of output dataset
    dst_kwds : dict, optional
        Dictionary of creation options and other parameters that will be
        overlaid on the profile of the output dataset.
    bar : bool, optional. 
        Whether to use a progress bar to display the merge progress. The default is False.


    Returns
    -------
    dest: numpy.ndarray
        Contents of all input rasters in single array
    out_transform: affine.Affine()
        Information for mapping pixel coordinates in `dest` to
        another coordinate system

    '''
    

    if method in MERGE_METHODS:
        copyto = MERGE_METHODS[method]
        
        if method == 'Quality':
            method_kwds = method_kwds.copy()
            Quality_sources = method_kwds.pop("Quality_sources")
            if len(Quality_sources) != len(sources):
                raise ValueError("Quality_sources 与 sources 数量不一致")
                
        
    elif callable(method):
        copyto = method
    else:
        raise ValueError(
            "Unknown method {}, must be one of {} or callable".format(
                method, list(MERGE_METHODS.keys())
            )
        )
    
    
    # Create a dataset_opener object
    dataset_opener = get_dataset_opener(sources[0])
    
    dst = None

    with ExitStack() as exit_stack:
        with dataset_opener(sources[0]) as first:
            first_profile = first.profile
            first_crs = first.crs
            best_res = first.res
            first_nodataval = first.nodatavals[0]
            nodataval = first_nodataval
            dt = first.dtypes[0]
            
            # 
            # src_count = first.count

            try:
                first_colormap = first.colormap(1)
            except ValueError:
                first_colormap = None
            
            
            #
            output_count = 1
            
            
            # Extent from option or extent of all inputs
            if bounds:
                dst_w, dst_s, dst_e, dst_n = bounds
            else:
                dst_w, dst_s, dst_e, dst_n = spatial_union(sources)
            
            # Resolution/pixel size
            if not res:
                res = best_res
            elif isinstance(res, numbers.Number):
                res = (res, res)
            elif len(res) == 1:
                res = (res[0], res[0])
            
            
            if target_aligned_pixels:
                dst_w = math.floor(dst_w / res[0]) * res[0]
                dst_e = math.ceil(dst_e / res[0]) * res[0]
                dst_s = math.floor(dst_s / res[1]) * res[1]
                dst_n = math.ceil(dst_n / res[1]) * res[1]
            
            
            # Compute output array shape. We guarantee it will cover the output
            # bounds completely
            output_width = int(round((dst_e - dst_w) / res[0]))
            output_height = int(round((dst_n - dst_s) / res[1]))

            output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(
                res[0], -res[1]
            )

            if dtype is not None:
                dt = dtype
                logger.debug("Set dtype: %s", dt)

            if nodata is not None:
                nodataval = nodata
                logger.debug("Set nodataval: %r", nodataval)
            
            
            inrange = False
            if nodataval is not None:
                # Only fill if the nodataval is within dtype's range
                if np.issubdtype(dt, np.integer):
                    info = np.iinfo(dt)
                    inrange = info.min <= nodataval <= info.max
                else:
                    if cmath.isfinite(nodataval):
                        info = np.finfo(dt)
                        inrange = info.min <= nodataval <= info.max
                        nodata_dt = np.min_scalar_type(nodataval)
                        inrange = inrange & np.can_cast(nodata_dt, dt)
                    else:
                        inrange = True
                
                #
                if not inrange:
                    warnings.warn(
                        f"Ignoring nodata value. The nodata value, {nodataval}, cannot safely be represented "
                        f"in the chosen data type, {dt}. Consider overriding it "
                        "using the --nodata option for better results. "
                        "Falling back to first source's nodata value."
                    )
                    nodataval = first_nodataval
            else:
                logger.debug("Set nodataval to 0")
                nodataval = 0
            
            
            # When dataset output is selected, we might need to create one
            # and will also provide the option of merging by chunks.
            dout_window = windows.Window(0, 0, output_width, output_height)
            if dst_path is not None:
                if isinstance(dst_path, DatasetWriter):
                    dst = dst_path
                else:
                    out_profile = first_profile
                    out_profile.update(**(dst_kwds or {}))
                    out_profile["transform"] = output_transform
                    out_profile["height"] = output_height
                    out_profile["width"] = output_width
                    out_profile["count"] = output_count
                    out_profile["dtype"] = dt
                    if nodata is not None:
                        out_profile["nodata"] = nodata
                    dst = rasterio.open(dst_path, "w", **out_profile)
                    exit_stack.enter_context(dst)


            chunk = dout_window
            
            # for chunk in chunks:
            dst_w, dst_s, dst_e, dst_n = windows.bounds(chunk, output_transform)
            dest = np.zeros((output_count, chunk.height, chunk.width), dtype=dt)
            if inrange:
                dest.fill(nodataval)
            
            if method == 'Quality':
                dest_qu = np.zeros((output_count, chunk.height, chunk.width), dtype=np.float32)
                dest_qu.fill(-np.inf)

            # From gh-2221
            chunk_bounds = windows.bounds(chunk, output_transform)
            chunk_transform = windows.transform(chunk, output_transform)

            dA, dB, QA, QB = None, None, None, None
            r = tqdm(sources) if bar else sources
            for idx, dataset in enumerate(r):
                
                
                with dataset_opener(dataset) as src:
                    
                    # Intersect source bounds and tile bounds
                    if first_crs != src.crs:
                        raise RastertoolError(f"CRS mismatch with source: {dataset}")

                    try:
                        ibounds = _intersect_bounds(
                            src.bounds, chunk_bounds, chunk_transform
                        )
                        sw = windows.from_bounds(*ibounds, src.transform)
                        cw = windows.from_bounds(*ibounds, chunk_transform)
                    except (ValueError, WindowError):
                        logger.info(
                            "Skipping source: src=%r, bounds=%r", src, src.bounds
                        )
                        continue



                    
                    
                    ### 加权平均
                    data = src.read(index,
                        # src_count
                        out_shape=(1, cw.height, cw.width),
                        masked=True,
                        window=sw,
                        resampling=resampling,
                    )
                    
                    maskB = data.mask
                    
                    
                    
                    if method == 'Quality':
                        QB = readarray(Quality_sources[idx],
                                       indexes=index,
                                       out_shape=(1, cw.height, cw.width),
                                       masked=True,
                                       window=sw,
                                       resampling=resampling)
                        dB = QB.filled(-np.inf).astype(np.float32)
                        # dB = QB
                    
                    else:
                        dB = distance_transform_edt(~maskB)
                    
                    
                    
                    cw = win_align(cw)
                    rows, cols = cw.toslices()
                    
                    
                    if cmath.isnan(nodataval):
                        maskA = np.isnan(dest)
                    elif not np.issubdtype(dest.dtype, np.integer):
                        maskA = np.isclose(dest, nodataval)
                    else:
                        maskA = dest == nodataval
                    # maskA = dest.mask   # 掩膜区域
                    
                    if method == 'Quality':
                        QA = dest_qu[:, rows, cols]
                        dA = QA
                    else:
                        dA_all = distance_transform_edt(~maskA)  # ~maskA有效值区域，有效值区域距离边界的欧式距离
                        dA = dA_all[:, rows, cols]
                    
                    
                    region = dest[:, rows, cols]
                    region_mask = maskA[:, rows, cols]
                    
                    
                    
                    copyto(region,
                           data,
                           region_mask,
                           data.mask,
                           dA, dB, **(method_kwds or {}))
                    
                    


            if dst:
                dw = windows.from_bounds(*chunk_bounds, output_transform)
                dw = win_align(dw)
                dst.write(dest, window=dw)

            if dst is None:
                if masked:
                    dest = np.ma.masked_equal(dest, nodataval, copy=False)
                return dest, output_transform
            else:
                if first_colormap:
                    dst.write_colormap(1, first_colormap)
                dst.close()
            
            




if __name__ == "__main__":

    from glob import glob
    
    
    buffer = 1000
    sampling = 30

    W = buffer * 2 / sampling   # 过渡带宽度
    sigma =  W / 3   # W ≈ 3σ
    os.chdir(r'F:/03dixinzu/02有机质/县域结果/有机质/pro')
    
    dir_tif = r'F:/03dixinzu/02有机质/县域结果/有机质/pro'
    
    
    out_ph = r'F:/03dixinzu/02有机质/县域结果/有机质/mosaic/抚州_func0x1.tif'
    
    tif_files = glob(os.path.join(dir_tif, '*.tif'))
    merge_weight(tif_files,method_kwds={'W':W, 'sigma':sigma},
                 dtype='float32',dst_path=out_ph,target_aligned_pixels=0,
                 bar=True)
    
    
    
    
    
    
    
    
    






















