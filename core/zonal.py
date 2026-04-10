"""
zonal.py — 栅格分区统计模块
============================
依赖: numpy, pandas, rasterio

无效值处理策略
--------------
使用 rasterio masked=True 读取掩膜数组（np.ma.MaskedArray）。
arr.mask == True 的像素直接视为无效，统一置 nan 后参与统计。
这样做的好处：
  1. rasterio 已综合处理 nodata / alpha 波段 / 内部掩膜等多种来源
  2. 避免浮点 nodata 数值比较的精度陷阱
  3. 无需用户手动传入 nodata 参数
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import Callable, Union

import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio
from rasterio.enums import Resampling

# ─────────────────────────────────────────────
# 类型别名
# ─────────────────────────────────────────────
RasterSource = Union[str, os.PathLike, rasterio.DatasetReader]
StatSpec = Union[str, Callable[[np.ndarray], float]]

# ─────────────────────────────────────────────
# 内置统计函数映射
# ─────────────────────────────────────────────
_BUILTIN_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean"   : np.nanmean,
    "sum"    : np.nansum,
    "max"    : np.nanmax,
    "min"    : np.nanmin,
    "std"    : np.nanstd,
    "var"    : np.nanvar,
    "median" : np.nanmedian,
    "count"  : lambda x: float(np.sum(~np.isnan(x))),   # 有效像素数
    "nodata" : lambda x: float(np.sum(np.isnan(x))),    # 无效像素数
    "range"  : lambda x: float(np.nanmax(x) - np.nanmin(x)),
    "p10"    : lambda x: np.nanpercentile(x, 10),
    "p25"    : lambda x: np.nanpercentile(x, 25),
    "p75"    : lambda x: np.nanpercentile(x, 75),
    "p90"    : lambda x: np.nanpercentile(x, 90),
    "p95"    : lambda x: np.nanpercentile(x, 95),
    "p99"    : lambda x: np.nanpercentile(x, 99),
}


def list_stats() -> list[str]:
    """返回所有内置统计量名称。"""
    return list(_BUILTIN_STATS.keys())


# ─────────────────────────────────────────────
# 底层 IO 工具
# ─────────────────────────────────────────────
def _get_dataset_opener(source: RasterSource):
    if isinstance(source, (str, os.PathLike)):
        return rasterio.open

    @contextmanager
    def _nullcontext(obj):
        try:
            yield obj
        finally:
            pass

    return _nullcontext


def readarray(
    source: RasterSource,
    indexes=None,
    out_shape=None,
    window=None,
    masked: bool = True,
    resampling: Resampling = Resampling.nearest,
    boundless: bool = False,
    fill_value=None,
) -> np.ma.MaskedArray:
    """
    读取栅格数据，默认返回 np.ma.MaskedArray。

    Parameters
    ----------
    source : str | PathLike | DatasetReader
        栅格文件路径或已打开的 rasterio 数据集。
    indexes : int | list[int], optional
        波段索引（rasterio 从 1 开始），默认读取全部波段。
    out_shape : tuple, optional
        输出形状，用于重采样读取。
    window : rasterio.windows.Window, optional
        读取窗口。
    masked : bool
        是否返回 MaskedArray，默认 True（推荐保持默认）。
        设为 False 时退化为普通 ndarray，无效值处理需自行处理。
    resampling : Resampling
        重采样方法，默认最近邻。
    boundless : bool
        是否允许越界读取。
    fill_value : scalar, optional
        越界填充值。

    Returns
    -------
    np.ma.MaskedArray
        arr.data : 原始数值数组
        arr.mask : True 表示无效像素（nodata / alpha / 内部掩膜）
    """
    opener = _get_dataset_opener(source)
    with opener(source) as src:
        arr = src.read(
            indexes,
            out_shape=out_shape,
            window=window,
            masked=masked,
            resampling=resampling,
            boundless=boundless,
            fill_value=fill_value,
        )
    return arr


def masked_to_float(arr: np.ma.MaskedArray) -> np.ndarray:
    """
    将 MaskedArray 转换为普通 float64 ndarray。
    被掩膜的无效像素（arr.mask == True）替换为 np.nan。

    这是无效值处理的核心转换。转换后，所有统计函数只需使用
    nan 安全版本（np.nanmean 等），无需再关心 nodata 数值。

    Parameters
    ----------
    arr : np.ma.MaskedArray
        readarray 返回的掩膜数组。

    Returns
    -------
    np.ndarray (float64)
        无效像素已替换为 nan 的普通数组。

    Notes
    -----
    使用 ma.getmaskarray 而非 arr.mask，确保 mask=False（标量，
    即无任何无效像素）时也能返回正确形状的布尔数组。
    """
    result = arr.astype(float).data.copy()
    result[ma.getmaskarray(arr)] = np.nan
    return result


# ─────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────
def _resolve_stats(
    stats: list[StatSpec],
) -> tuple[list[str], list[Callable[[np.ndarray], float]]]:
    """将 stats 列表解析为 (名称列表, 函数列表)。"""
    names, funcs = [], []
    for s in stats:
        if isinstance(s, str):
            key = s.lower()
            if key not in _BUILTIN_STATS:
                raise ValueError(
                    f"不支持的统计类型: '{s}'。"
                    f"内置支持: {list(_BUILTIN_STATS.keys())}，"
                    f"或传入可调用对象。"
                )
            names.append(s)
            funcs.append(_BUILTIN_STATS[key])
        elif callable(s):
            names.append(getattr(s, "__name__", repr(s)))
            funcs.append(s)
        else:
            raise TypeError(
                f"stats 元素必须是字符串或可调用对象，得到: {type(s)}"
            )
    return names, funcs


def _is_nan_area(area) -> bool:
    """判断分区值是否代表 NaN 分区（None 或 float NaN）。"""
    if area is None:
        return True
    try:
        return bool(np.isnan(area))
    except (TypeError, ValueError):
        return False


def _unique_areas(dst_arr: np.ndarray, dst_nan_mask: np.ndarray) -> list:
    """
    从分区数组提取唯一分区值。

    Parameters
    ----------
    dst_arr : np.ndarray
        分区数组（float64，无效位置已为 nan）。
    dst_nan_mask : np.ndarray
        分区无效像素掩膜（True 表示无效）。

    Returns
    -------
    list
        唯一分区值列表；若存在无效分区像素则末尾追加 np.nan。
    """
    valid_vals = dst_arr[~dst_nan_mask]
    unique_vals = np.unique(valid_vals).tolist()
    if np.any(dst_nan_mask):
        unique_vals.append(np.nan)
    return unique_vals


def _compute_row(
    values: np.ndarray,
    stat_funcs: list[Callable],
) -> list[float]:
    """
    对一组像素值计算所有统计量。
    values 已为 float64（含 nan），stat_funcs 均为 nan 安全函数。
    """
    if values.size == 0:
        return [np.nan] * len(stat_funcs)
    row = []
    for fn in stat_funcs:
        try:
            row.append(float(fn(values)))
        except Exception:
            row.append(np.nan)
    return row


# ─────────────────────────────────────────────
# 核心：单波段分区统计
# ─────────────────────────────────────────────
def zonal(
    raster_in: RasterSource,
    dst_in: RasterSource,
    stats: list[StatSpec],
    areas: list | None = None,
    dic: dict | None = None,
    index: str | list[str] | None = "area",
) -> pd.DataFrame:
    """
    栅格分区统计（单波段）。

    对 ``raster_in`` 中每个像素，按 ``dst_in`` 中对应的分区编号归组，
    对每组计算指定统计量。

    无效值处理
    ----------
    两个栅格均使用 ``masked=True`` 读取（MaskedArray）：

    - 值栅格：``arr.mask == True`` 的像素置 nan，统计时自动跳过。
    - 分区栅格：``arr.mask == True`` 的像素不参与任何分区（作为 NaN
      分区单独统计，仅当 areas 为 None 且存在此类像素时出现在结果中）。

    无需手动传入 nodata 数值，rasterio 已在内部综合处理：
    文件元数据中的 nodata、alpha 波段、内部掩膜（TIFF internal mask）等。

    Parameters
    ----------
    raster_in : str | PathLike | DatasetReader
        输入值栅格（单波段）。
    dst_in : str | PathLike | DatasetReader
        分区栅格，应为整型栅格。浮点型分区栅格会触发警告。
    stats : list[str | callable]
        统计量列表。内置支持：
        'mean', 'sum', 'max', 'min', 'std', 'var', 'median',
        'count', 'nodata', 'range', 'p10', 'p25', 'p75', 'p90', 'p95', 'p99'。
        也可传入接受 1D float ndarray、返回标量的任意可调用对象。
    areas : list, optional
        需要统计的分区值。为 None 时统计所有分区，默认 None。
    dic : dict, optional
        分区值到标签的映射，用于替换结果索引的显示名称。
        例如 {1: '耕地', 2: '林地'}。
    index : str | list[str] | None
        结果 DataFrame 的索引列名。默认 'area'（分区值列）。
        传入 None 则保留默认整数索引。

    Returns
    -------
    pd.DataFrame
        行为各分区，列为各统计量。

    Raises
    ------
    AssertionError
        stats 不是 list 或 tuple。
    ValueError
        stats 中包含不支持的字符串统计量。
    TypeError
        stats 中元素类型错误。

    Notes
    -----
    两个栅格的 CRS、分辨率、范围须一致，否则结果无意义。

    Examples
    --------
    >>> df = zonal('value.tif', 'zone.tif', ['mean', 'sum', 'count'])
    >>> df = zonal('value.tif', 'zone.tif', ['mean', 'std'],
    ...            dic={1: '耕地', 2: '林地'})
    >>> df = zonal('value.tif', 'zone.tif',
    ...            ['mean', lambda x: np.nanpercentile(x, 95)])
    """
    assert isinstance(stats, (list, tuple)), "请保证 stats 是一个 list 或 tuple"
    stat_names, stat_funcs = _resolve_stats(list(stats))

    # ── 读取掩膜数组 → 转为 float（无效像素为 nan）────────────────
    src_arr = masked_to_float(readarray(source=raster_in, masked=True)).ravel()
    dst_arr = masked_to_float(readarray(source=dst_in,    masked=True)).ravel()

    # 分区无效掩膜：dst 被掩膜的像素不属于任何有效分区
    dst_nan_mask = np.isnan(dst_arr)

    # ── 确定分区列表 ──────────────────────────────────────────────
    if areas is None:
        areas = _unique_areas(dst_arr, dst_nan_mask)

    if len(areas) >= 1000:
        warnings.warn(
            f"\n分区数为 {len(areas)}，分区栅格可能为浮点型栅格，"
            "建议确认 dst_in 是否为整型。",
            UserWarning,
            stacklevel=2,
        )

    dic = dic or {}

    # ── 逐分区统计 ────────────────────────────────────────────────
    rows: list[list[float]] = []
    area_labels: list = []

    for area in areas:
        if _is_nan_area(area):
            # NaN 分区：dst 本身为无效像素的位置
            zone_mask = dst_nan_mask
            label = np.nan
        else:
            # 正常分区：dst == area 且 dst 本身有效
            zone_mask = (dst_arr == float(area)) & ~dst_nan_mask
            label = area

        # src_arr 中对应像素（保留其自身的 nan，统计函数会自动跳过）
        values = src_arr[zone_mask]
        rows.append(_compute_row(values, stat_funcs))
        area_labels.append(dic.get(label, label))

    # ── 组装 DataFrame ────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=stat_names)
    df.insert(0, "area", area_labels)

    if index:
        df.set_index(index, drop=True, inplace=True)

    return df


# ─────────────────────────────────────────────
# 扩展：多波段分区统计
# ─────────────────────────────────────────────
def zonal_multiband(
    raster_in: RasterSource,
    dst_in: RasterSource,
    stats: list[StatSpec],
    areas: list | None = None,
    dic: dict | None = None,
    index: str | list[str] | None = "area",
    band_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    多波段栅格分区统计。

    对每个波段分别进行分区统计，返回以波段名为键的字典。
    分区栅格只读取一次，各分区的 zone_mask 预先计算并复用。

    Parameters
    ----------
    raster_in : str | PathLike | DatasetReader
        输入多波段栅格。
    dst_in : str | PathLike | DatasetReader
        分区栅格。
    stats : list[str | callable]
        统计量列表，同 ``zonal``。
    areas : list, optional
        需要统计的分区值，同 ``zonal``。
    dic : dict, optional
        分区值到标签映射，同 ``zonal``。
    index : str | list[str] | None
        结果 DataFrame 索引列，同 ``zonal``。
    band_names : list[str], optional
        各波段名称列表，长度须与波段数一致。
        为 None 时使用 'band_1', 'band_2', ... 命名。

    Returns
    -------
    dict[str, pd.DataFrame]
        键为波段名称，值为对应的分区统计 DataFrame。

    Examples
    --------
    >>> results = zonal_multiband('multi.tif', 'zone.tif', ['mean', 'std'])
    >>> results['band_1']
    """
    assert isinstance(stats, (list, tuple)), "请保证 stats 是一个 list 或 tuple"
    stat_names, stat_funcs = _resolve_stats(list(stats))

    # ── 分区栅格只读一次 ──────────────────────────────────────────
    dst_arr = masked_to_float(readarray(source=dst_in, masked=True)).ravel()
    dst_nan_mask = np.isnan(dst_arr)

    _areas = areas if areas is not None else _unique_areas(dst_arr, dst_nan_mask)

    if len(_areas) >= 1000:
        warnings.warn(
            f"\n分区数为 {len(_areas)}，分区栅格可能为浮点型栅格。",
            UserWarning,
            stacklevel=2,
        )

    dic = dic or {}

    # ── 预计算各分区的 zone_mask（所有波段共用）──────────────────
    zone_info: list[tuple[np.ndarray, object]] = []
    for area in _areas:
        if _is_nan_area(area):
            zone_info.append((dst_nan_mask, np.nan))
        else:
            zone_info.append(
                ((dst_arr == float(area)) & ~dst_nan_mask, area)
            )

    area_labels = [dic.get(label, label) for _, label in zone_info]

    # ── 逐波段读取并统计 ──────────────────────────────────────────
    opener = _get_dataset_opener(raster_in)
    results: dict[str, pd.DataFrame] = {}

    with opener(raster_in) as src:
        n_bands = src.count
        if band_names is None:
            band_names = [f"band_{i+1}" for i in range(n_bands)]
        else:
            if len(band_names) != n_bands:
                raise ValueError(
                    f"band_names 长度 ({len(band_names)}) 与波段数 ({n_bands}) 不一致"
                )

        for i, bname in enumerate(band_names, start=1):
            band_arr = masked_to_float(src.read(i, masked=True)).ravel()

            rows = [
                _compute_row(band_arr[zone_mask], stat_funcs)
                for zone_mask, _ in zone_info
            ]

            df = pd.DataFrame(rows, columns=stat_names)
            df.insert(0, "area", area_labels)
            if index:
                df.set_index(index, drop=True, inplace=True)
            results[bname] = df

    return results


# ─────────────────────────────────────────────
# 便捷函数
# ─────────────────────────────────────────────
def zonal_to_csv(
    raster_in: RasterSource,
    dst_in: RasterSource,
    stats: list[StatSpec],
    out_path: str | os.PathLike,
    **kwargs,
) -> pd.DataFrame:
    """
    执行分区统计并将结果保存为 CSV 文件。

    Parameters
    ----------
    raster_in, dst_in, stats
        同 ``zonal``。
    out_path : str | PathLike
        输出 CSV 文件路径。
    **kwargs
        其余参数透传给 ``zonal``。

    Returns
    -------
    pd.DataFrame
        统计结果（同时写入 CSV）。
    """
    df = zonal(raster_in, dst_in, stats, **kwargs)
    df.to_csv(out_path)
    return df