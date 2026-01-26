# -*- coding: utf-8 -*-


import os
import numbers
import math
import cmath
import warnings
import logging
import numpy as np

from contextlib import ExitStack
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from rastertool.functions import get_dataset_opener#, out
from rastertool.errors import (
                                MergeError,
                                RastertoolError,
                                WindowError,
                                )


import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetWriter
from rasterio import windows
from rasterio.transform import Affine
# from rasterio.windows import subdivide

logger = logging.getLogger(__name__)

def spatial_union(sources):
    '''Calculate the union of the spaces and return the bounds'''
    
    dataset_opener = get_dataset_opener(sources[0])
    xs = []
    ys = []

    for i, dataset in enumerate(sources):
        with dataset_opener(dataset) as src:
            src_transform = src.transform



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



#From rasterio.merge
def _copy_first(merged_data, new_data, merged_mask, new_mask, **kwargs):
    """Returns the first available pixel."""
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def copy_Gaussian(merged_data, new_data,
                  validAB,
                  dA, dB,
                  W=None,
                  sigma=None,
                  eps=1e-6):


    # 计算最大距离（仅重叠区）
    if W is None:
        dmax_A = np.max(dA[validAB])
        dmax_B = np.max(dB[validAB])
        W = max(dmax_A, dmax_B)
    else:
        dmax_A = dmax_B = W

    if sigma is None:
        sigma = W / 3.0
    sigma = max(sigma, eps)  # 防止小于等于0

    # 初始化权重为 0
    wA = np.zeros_like(dA, dtype=np.float32)
    wB = np.zeros_like(dB, dtype=np.float32)

    # 仅在重叠区计算高斯权重
    xA = np.maximum(dmax_A - dA[validAB], 0)   # 防止小于0
    xB = np.maximum(dmax_B - dB[validAB], 0)
    
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    wA[validAB] = np.exp(-(xA * xA) * inv_2sigma2)
    wB[validAB] = np.exp(-(xB * xB) * inv_2sigma2)

    # 加权融合
    V = (wA * merged_data + wB * new_data) / (wA + wB + eps)

    np.copyto(merged_data, V, where=validAB, casting="unsafe")

def copy_IDW(merged_data, new_data,
             validAB,
             dA, dB,
             W=None,
             k=2,
             eps=1e-6):


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
    denomA = np.maximum(dmax_A - dA[validAB], eps)   # 防止小于等于0
    denomB = np.maximum(dmax_B - dB[validAB], eps)

    wA[validAB] = 1.0 / (denomA ** k)
    wB[validAB] = 1.0 / (denomB ** k)

    # 加权融合（只在重叠区）
    V = (wA * merged_data + wB * new_data) / (wA + wB + eps)

    np.copyto(merged_data, V, where=validAB, casting="unsafe")


def copy_Voronoi(merged_data, new_data,
                 validAB,
                 dA, dB, **kwargs):
    
    choose_new = (dB > dA) & validAB

    np.copyto(merged_data, new_data, where=choose_new, casting="unsafe")



    
# def copy_Average(merged_data, new_data,
#              merged_mask, new_mask,
#              wA=1, wB=1,
#              **kwargs
#              ):
#     validAB = (~merged_mask) & (~new_mask)
    
#     V = (wA*merged_data + wB*new_data) / (wA + wB)
    
#     np.copyto(merged_data, new_data, where=merged_mask, casting="unsafe")
#     np.copyto(merged_data, V, where=validAB, casting="unsafe")




MERGE_METHODS = {
    "Gaussian": copy_Gaussian,
    "IDW": copy_IDW,
    "Voronoi": copy_Voronoi,
    # "Average": copy_Average,

}


def merge_distance_weight(sources,          # 输入栅格文件列表
                          method='Gaussian', # 加权方法
                          method_kwds=None,  # 方法参数
                          index=1,           # 处理的波段序号
                          bounds=None,       # 输出范围
                          res=None,          # 输出分辨率
                          nodata=None,       # 空值
                          dtype=None,        # 数据类型
                          resampling=Resampling.nearest, # 重采样方法
                          target_aligned_pixels=True,    # 像素对齐首幅影像
                          masked=False,      # 是否返回掩码数组
                          dst_path=None,     # 输出路径，存在输出路径则无返回值
                          dst_kwds=None,     # 输出参数
                          bar_name=None):         # 进度条名称与是否显示进度条

    """
    Merge multiple raster datasets into a single raster using weighted
    blending in overlapping areas.
    在重叠区域使用加权融合方式，将多个栅格数据集合并为单一栅格。

    This function is based on ``rasterio.merge.merge`` but extends it by
    introducing several weighting strategies to smoothly blend pixel
    values in overlapping regions. It is especially suitable for mosaicking
    raster tiles with buffer zones to avoid visible seams.
    本函数基于 ``rasterio.merge.merge`` 实现，并扩展了多种加权策略，
    用于在重叠区域平滑融合像元值，特别适合带缓冲区的栅格块拼接，
    以避免明显接缝。

    Parameters
    ----------
    sources : list
        Sequence of input raster datasets opened in read mode or path-like
        objects pointing to raster files.
        输入栅格数据集对象（读模式）或栅格文件路径列表。

    method : str or callable, optional. Default: ``"Gaussian"``
        Weighting method used to resolve pixel values in overlapping areas.
        重叠区域像元值融合所使用的加权方法。

        Built-in methods:
        内置方法：

        - ``"Gaussian"`` : Gaussian distance-based attenuation.
          高斯距离衰减加权。
        - ``"IDW"`` : Inverse Distance Weighting.
          反距离加权法。
        - ``"Voronoi"`` : Select the value from the raster whose valid region
          is closer to the pixel.
          选择距离该像元最近的栅格数据值（泰森多边形思想）。
        - ``"Average"`` : Pixel-wise weighted average of existing and new data.
          对新旧数据进行逐像元加权平均。

        A custom callable can also be provided with the following signature:
        也可以提供自定义函数，函数签名如下::

            method(merged_data, new_data,
                   validAB,
                   dA, dB, **method_kwds)

        where:
        参数说明：

        - ``merged_data`` : ndarray
            Existing output array to be updated.
            当前输出数组（待更新）。
            
        - ``new_data`` : ndarray
            New raster data to merge, same shape as ``merged_data``.
            新读入的栅格数据，与 merged_data 形状相同。
            
            
          ``validAB`` : ndarray of bool
            Overlapping valid region mask of merged_data and new_data.
            merged_data 与 new_data 均为有效值的重叠区域掩码（True 表示参与加权融合的像元）。
            
        - ``dA`` : ndarray
            Euclidean distance to nearest invalid pixel in ``merged_data``.
            merged_data 中有效像元到边界的欧氏距离。
            
        - ``dB`` : ndarray
            Euclidean distance to nearest invalid pixel in ``new_data``.
            new_data 中有效像元到边界的欧氏距离。
            
        - ``**method_kwds`` : 
            Additional parameters for the weighting method.
            加权方法所需的附加参数。
            
        - ``Notes``
            -----
            - Only pixels where ``validAB`` is True should be blended.
              只应对 validAB 为 True 的像素进行融合处理。
            - Before calling this function, the pixels that were invalid in the merged_data but valid in the new_data have already been filled.
              在调用此函数之前，那些 merged_data 无效但 new_data 有效的像素已被填充。
            - This function must modify ``merged_data`` in-place and should not return anything.
              此函数必须直接修改``merged_data``这一变量，并且不应返回任何值。



    method_kwds : dict, optional .Default is ``None``.
        Parameters required for the selected weighting method.
        所选加权方法所需的参数字典。


        The following keys are supported for each built-in method:
        各内置方法支持的参数如下：

        **Gaussian**

        W : float, optional. Default is ``None``.
            Width of the overlapping zone in number of pixels.
            重叠区域宽度（以像元数表示）。

            If ``None``, the maximum distance in the overlapping area is
            computed from the data and used as ``W``.
            若为 ``None``，则自动计算重叠区最大距离作为 W。

            Otherwise, the provided value is used.
            否则使用用户提供的值。

        sigma : float, optional. Default is ``None``.
            Standard deviation of the Gaussian function, controlling the
            decay rate (smaller values result in faster decay).
            高斯函数标准差，用于控制权重衰减速度（越小衰减越快）。

            If ``None``, ``sigma = W / 3``.
            若为 ``None``，则 ``sigma = W / 3``。

            Example: buffer width ``w = 1000 m``, resolution ``res = 30 m``::
            示例：缓冲区宽度 w=1000 米，分辨率 res=30 米::

                W = 2 * w / res
                sigma = W / 3

        **IDW**

        W : float, optional. Default is ``None``.
            Width of the overlapping zone in number of pixels.
            重叠区域宽度（像元数）。

            Same behavior as Gaussian method.
            与 Gaussian 方法中 W 的含义相同。

        k : float, optional. Default is ``2``.
            Distance decay parameter controlling how fast the weight
            decreases with distance. Default is ``2``.
            距离衰减参数，控制权重随距离减小的速度，默认值为 2。

        **Voronoi**

        No additional parameters.
        无额外参数。


    index : int, optional. Default is ``1``.
        Band index to read from each input raster
        读取的波段序号，默认第 1 波段。

    bounds : tuple of float, optional. Default is ``None``.
        Bounds of the output raster in the form ``(left, bottom, right, top)``.
        输出栅格的空间范围（左、下、右、上）。

        If not provided, the spatial union of all inputs is used.
        若不提供，则自动计算所有输入栅格的联合范围。

    res : float or tuple of float, optional. Default is ``None``.
        Output resolution in CRS units.
        输出分辨率（坐标系单位）。

        If a single value is given, square pixels are assumed.
        若提供单值，则像元为正方形。

        If not provided, the resolution of the first raster is used.
        默认使用第一个输入栅格的分辨率。

    nodata : float, optional. Default is ``None``.
        NoData value used in the output raster.
        输出栅格使用的无效值。

        Defaults to the NoData value of the first raster.
        默认使用第一个栅格的 NoData。

    dtype : numpy.dtype or str, optional. Default is ``None``.
        Data type of the output raster.
        输出栅格的数据类型。

        Defaults to the first raster's dtype.
        默认与第一个栅格一致。

    resampling : rasterio.enums.Resampling, optional. Default is ``Resampling.nearest``.
        Resampling method used when reading input rasters.
        读取输入栅格时使用的重采样方法。

        

    target_aligned_pixels : bool, optional.Default is True.
        If True, align output bounds to integer multiples of pixel size
        (GDAL ``-tap`` behavior).
        若为 True，则输出像元边界按像元大小整数倍对齐（等同 GDAL 的 -tap）。

        

    masked : bool, optional. Default is False.
        If True, return a masked array.
        若为 True，则返回掩码数组。

        

    dst_path : str or PathLike or DatasetWriter, optional. Default is ``None``.
        Output file path or open Rasterio dataset.
        输出文件路径或已打开的 Rasterio 数据集。

        If None, the merged array is returned.
        若为 None，则仅返回合并结果数组。

    dst_kwds : dict, optional. Default is ``None``.
        Additional profile options for output dataset.
        输出数据集的附加创建参数。

    bar_name : str, optional. Default is ``None``.
        Progress bar name
        进度条名称
        Whether to display a progress bar.
        是否显示进度条。
        It is only when the value is None that it will not be displayed.
        只有在None时不显示


    Returns
    -------
    dest : ndarray or MaskedArray
        Merged raster array.
        合并后的栅格数组。

    out_transform : affine.Affine
        Affine transform of the output raster.
        输出栅格的仿射变换参数。

    Notes
    -----
    - All input rasters must share the same CRS.
      所有输入栅格必须使用相同坐标参考系。
    - Only rectilinear north-up rasters are supported.
      仅支持正北向、无旋转栅格。
    - Weighted blending is applied only in overlapping valid areas.
      仅在双方均有有效值的重叠区域执行加权融合。

    Examples
    --------
    Gaussian weighted merge::

        merge_weight(
            sources,
            method="Gaussian",
            method_kwds={"W": 100, "sigma": 33},
            dst_path="mosaic.tif"
        )

    IDW weighted merge::

        merge_weight(
            sources,
            method="IDW",
            method_kwds={"k": 2},
            dst_path="mosaic_idw.tif"
        )
    """

    

    if method in MERGE_METHODS:
        copyto = MERGE_METHODS[method]
        
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
            


            # From gh-2221
            chunk_bounds = windows.bounds(chunk, output_transform)
            chunk_transform = windows.transform(chunk, output_transform)

            # run
            if bar_name is None:
                r = sources
            elif isinstance(bar_name, str):
                r = tqdm(sources, desc=bar_name)
            else:
                r = tqdm(sources)
                
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
                        sw = win_align(sw)
                        cw = windows.from_bounds(*ibounds, chunk_transform)
                        cw = win_align(cw)
                    except (ValueError, WindowError):
                        logger.info(
                            "Skipping source: src=%r, bounds=%r", src, src.bounds
                        )
                        continue
                    
                    ###  主要修改及操作部分 ###
                    #  待merge栅格B
                    data = src.read([index],
                                    out_shape=(1, cw.height, cw.width),
                                    masked=True,
                                    window=sw,
                                    resampling=resampling,
                                    )
                    
                    
                    maskB = data.mask

                    
                    rows, cols = cw.toslices()
                    
                    # 掩膜判定
                    if cmath.isnan(nodataval):
                        maskA = np.isnan(dest)
                    elif not np.issubdtype(dest.dtype, np.integer):
                        maskA = np.isclose(dest, nodataval)
                    else:
                        maskA = dest == nodataval
                    
                    
                    region = dest[:, rows, cols]
                    region_mask = maskA[:, rows, cols]
                    
                    
                    
                    # 先将不重叠的部分合并
                    _copy_first(region,
                                data,
                                region_mask,
                                data.mask)
                    # 是否有重叠，如无则跳过
                    validAB = (~region_mask) & (~maskB)
                    if not np.any(validAB):
                        continue
                    
                    ### 重叠区域加权平均
                    dA_all = distance_transform_edt(~maskA)  # (~maskA)有效值区域，有效值区域距离边界的欧式距离
                    dA = dA_all[:, rows, cols]
                    dB = distance_transform_edt(~maskB)  # (~maskB)有效值区域，有效值区域距离边界的欧式距离
                    
                    
                    
                    ### 调用加权平均函数,copyto会将结果写入dest
                    copyto(region,
                           data,
                           validAB,
                           dA, dB, **(method_kwds or {}))
                    
                    ######################################

            ## output
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
    
    
    '''试验代码位于rastertool/tast/merge/抚州拼接.py'''
    from glob import glob
    
    
    buffer = 1000
    sampling = 30

    W = buffer * 2 / sampling   # 过渡带宽度
    sigma =  W / 3   # W ≈ 3σ
    os.chdir(r'F:/03dixinzu/02有机质/县域结果/有机质/pro')
    
    dir_tif = r'F:/03dixinzu/02有机质/县域结果/有机质/pro'
    
    
    out_ph = r'F:/03dixinzu/02有机质/县域结果/有机质/mosaic/抚州_Voronoi.tif'
    
    tif_files = glob(os.path.join(dir_tif, '*.tif'))
    method_kwds = None
    method = 'Voronoi'
    method = 'IDW'
    out_ph = r'F:/03dixinzu/02有机质/县域结果/有机质/mosaic/抚州_%s.tif' % method
    
    
    
    merge_distance_weight(tif_files,
                          method=method,
                          method_kwds=method_kwds,
                          dtype='float32',dst_path=out_ph,target_aligned_pixels=0,
                          bar=True)
    
    # dir_d = r'F:/03dixinzu/02有机质/县域结果/有机质/d'
    # for tif in tif_files:
    #     name = os.path.basename(tif)
    #     out_ph = os.path.join(dir_d, name)
        
    #     with rasterio.open(tif) as src:
    #         arr = src.read(masked=True)
    #         mask = arr.mask
    #         d = distance_transform_edt(~mask)
    #         profile = src.profile
    #         profile.update({'nodata':np.nan, 'dtype':'float32', 'compress':'lzw'})
        
    #     out(out_ph, d, profile)
        
        
    
    
    
    
    
    






















