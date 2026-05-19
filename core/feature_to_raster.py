"""
矢量转栅格并生成栅格属性表 (RAT)
依赖: geopandas, rasterio, numpy, pandas, shapely
"""



import os
from typing import Union, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from pandas.api.types import is_numeric_dtype
from rastertool.functions import read, get_dataset_opener
from rastertool.core.unify import unify
from rastertool.core.clip import clip
from rasterio.warp import reproject
from rasterio.mask import mask
from shapely.geometry import mapping, box
def bounds_to_mask_shapes(bounds):
    """
    将 bounds 转成可用于 rasterio.mask.mask 的 shapes

    参数
    ----------
    bounds : list | tuple
        [left, bottom, right, top]

    返回
    -------
    shapes : list
        可直接传给 rasterio.mask.mask 的 shapes
    """
    left, bottom, right, top = bounds
    geom = box(left, bottom, right, top)
    return [mapping(geom)]

def feature_to_raster(
    source: Union[str, gpd.GeoDataFrame],
    out_dest: str,
    pixel: float,
    field: Optional[str] = None,
    crs=None,
    bounds=None,
    dtype=None,
    nodata=None,
    rat: bool = True,
    rat_type: str = 'dbf',
) -> Optional[pd.DataFrame]:
    """
    将矢量要素栅格化为 GeoTIFF，并可选生成栅格属性表（RAT）。

    参数
    ----------
    source : str | geopandas.GeoDataFrame
        输入矢量数据。可为矢量文件路径，或已读取的 GeoDataFrame。
        若为文件路径，将使用 geopandas.read_file() 读取。
    
    out_dest : str
        输出栅格文件路径，通常为 .tif。

    pixel : float
        栅格像元大小。单位与数据坐标系单位一致。
        例如投影坐标系下通常为米，地理坐标系下通常为度。

    field : str, optional
        用于栅格赋值的字段名。
        - 若为 None，则按要素顺序赋整数值 0, 1, 2, ...
        - 若字段为数值型，则直接使用字段值写入栅格
        - 若字段为分类型/字符串型，则自动编码为整数类别值，并生成类别映射表

    crs : any, optional
        目标坐标参考系。
        - 若指定，则输入数据会重投影到该 CRS 后再栅格化
        - 若未指定，则要求输入数据本身必须有 CRS
        - 若输入无 CRS 且 crs=None，则抛出异常

    dtype : str, optional
        输出栅格数据类型，例如：
        'int16', 'int32', 'uint8', 'float32' 等。
        若不指定，将自动推断：
        - field=None 时默认为 int32
        - 数值整数字段默认为 int32
        - 数值浮点字段默认为 float32
        - 分类字段默认为 int32

    nodata : int | float, optional
        输出栅格 NoData 值。
        若不指定，将自动推断：
        - 整数类型：取对应整数类型最小值
        - 浮点类型：使用 np.nan

    rat : bool, default True
        是否生成栅格属性表（Raster Attribute Table, RAT）。
        仅当输出栅格为整数类型时有效；
        若输出为浮点型，即使 rat=True 也不会生成属性表。

    rat_type : str, default 'dbf'
        属性表输出格式。
        支持：
        - 'dbf'：输出 ArcGIS 风格 VAT 表，文件名为 *.tif.vat.dbf
        - 'csv'：输出 CSV 表
        - 其他值：默认按 CSV 输出

    返回
    -------
    pandas.DataFrame | None
        当成功生成 RAT 时，返回属性表 DataFrame；
        若 rat=False 或输出栅格不是整数类型，则返回 None。

        返回表通常包含以下字段：
        - FID   : 记录编号
        - Value : 栅格像元值
        - Count : 对应像元数量
        - field : 原分类字段值（仅分类字段时存在）

    功能说明
    ----------
    1. 读取输入矢量数据
       支持文件路径和 GeoDataFrame 两种输入形式。

    2. 处理坐标系
       若指定 crs，则重投影后再栅格化；否则要求输入已有坐标系。

    3. 根据矢量总范围计算输出栅格大小
       使用 gdf.total_bounds 获取边界，
       并结合 pixel 计算宽度、高度以及仿射变换参数。

    4. 生成栅格烧录值（burn values）
       - field=None：按要素顺序赋值
       - 数值字段：直接写值
       - 字符串/分类字段：先编码为整数，再保留映射关系

    5. 自动推断输出 dtype
       根据字段类型自动选择整数或浮点栅格类型。

    6. 处理 NoData
       将 burn_vals 中的空值统一替换为 nodata。

    7. 执行栅格化
       使用 rasterio.features.rasterize() 将要素写入栅格数组。

    8. 写出 GeoTIFF
       使用 rasterio 保存为单波段 GeoTIFF，并启用 LZW 压缩。

    9. 生成 RAT
       对非 NoData 像元统计唯一值及其数量，并与原始值映射表合并。

    10. 保存 RAT
        - dbf：保存为 .tif.vat.dbf
        - csv：保存为同名 .csv

    异常
    ----------
    FileNotFoundError
        当 source 为文件路径且文件不存在时抛出。

    ValueError
        当输入 GeoDataFrame 为空时抛出。
        当输入数据缺失 CRS 且未指定 crs 时抛出。
        当指定的 field 不存在时抛出。

    注意事项
    ----------
    1. 分类字段会被编码为整数值写入栅格，原始类别名保存在 RAT 中。
    2. 浮点型栅格不会生成 RAT，因为属性表通常仅适用于整数分类栅格。
    3. 若 rat_type='dbf' 但未安装 dbf 库，将自动退回为 CSV 输出。
    4. 输出栅格范围严格依据输入矢量总边界计算，不额外留边。
    5. 多个要素重叠时，后出现的要素值会覆盖先前要素值。



    示例
    ----------
    1. 按要素编号栅格化
    >>> feature_to_raster("landuse.shp", "landuse.tif", pixel=30)

    2. 按数值字段栅格化
    >>> feature_to_raster("elevation_zone.shp", "zone.tif", pixel=50, field="class_id")

    3. 按分类字段栅格化并输出 CSV 属性表
    >>> rat_df = feature_to_raster(
    ...     "soil.shp",
    ...     "soil.tif",
    ...     pixel=100,
    ...     field="soil_type",
    ...     rat=True,
    ...     rat_type="csv"
    ... )

    4. 指定目标坐标系
    >>> feature_to_raster(
    ...     source=gdf,
    ...     out_dest="output.tif",
    ...     pixel=10,
    ...     field="type",
    ...     crs="EPSG:4547"
    ... )
    """

    # ---------------------------
    # 1. 读取数据
    # ---------------------------
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        gdf = gpd.read_file(source)
    else:
        gdf = source.copy()

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty")

    # ---------------------------
    # 2. CRS 处理
    # ---------------------------
    if crs is not None:
        gdf = gdf.to_crs(crs)
    elif gdf.crs is None:
        raise ValueError("Input has no CRS and crs not specified")

    crs = gdf.crs

    # ---------------------------
    # 3. bounds → 栅格尺寸
    # ---------------------------
    minx, miny, maxx, maxy = gdf.total_bounds

    width = int(np.ceil((maxx - minx) / pixel))
    height = int(np.ceil((maxy - miny) / pixel))

    transform = from_origin(minx, maxy, pixel, pixel)

    # ---------------------------
    # 4. field → burn 值
    # ---------------------------
    if field is None:
        burn_vals = np.arange(len(gdf), dtype="int32")
        value_map = pd.DataFrame({
            "FID": burn_vals,
            "Value": burn_vals
        })

    else:
        if field not in gdf.columns:
            raise ValueError(f"Field '{field}' not found")

        col = gdf[field]

        # ===== 数值字段 =====
        if is_numeric_dtype(col):
            burn_vals = col.to_numpy()

            value_map = pd.DataFrame({
                "FID": np.arange(len(col)),
                "Value": burn_vals
            })

        # ===== 分类字段 =====
        else:
            col = col.astype("string")

            # 保留非空类别
            cats = pd.Series(col.dropna().unique()).sort_values()

            cat_map = {v: i for i, v in enumerate(cats)}

            burn_vals = col.map(cat_map).to_numpy()

            value_map = pd.DataFrame({
                "Value": list(cat_map.values()),
                field: list(cat_map.keys())
            })

    # ---------------------------
    # 5. dtype 推断
    # ---------------------------
    if dtype is None:
        if field is None:
            dtype = "int32"
        else:
            if is_numeric_dtype(burn_vals):
                if np.issubdtype(np.array(burn_vals).dtype, np.integer):
                    dtype = "int32"
                else:
                    dtype = "float32"
            else:
                dtype = "int32"

    dtype_np = np.dtype(dtype)

    # ---------------------------
    # 6. nodata 处理
    # ---------------------------
    if nodata is None:
        if np.issubdtype(dtype_np, np.integer):
            nodata = np.iinfo(dtype_np).min
        else:
            nodata = np.nan

    # 替换 NaN → nodata（关键！）
    burn_vals = np.where(pd.isna(burn_vals), nodata, burn_vals)

    # ---------------------------
    # 7. rasterize
    # ---------------------------
    shapes = (
        (geom, val)
        for geom, val in zip(gdf.geometry, burn_vals)
        if geom is not None
    )

    _raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=dtype,
        masked=True
    )
    raster = _raster.data
    raster_mask = _raster.mask

    # ---------------------------
    # 8. 写出 raster
    # ---------------------------
    # if bounds is None:
    with rasterio.open(
        out_dest,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(raster, 1)
    # else:
    #     shape = bounds_to_mask_shapes(bounds)
    #     mask()
    
    
    
    
    
    

    # ---------------------------
    # 9. RAT 生成
    # ---------------------------
    if not rat:
        return None

    if not np.issubdtype(dtype_np, np.integer):
        return None

    # 像元统计
    
    valid = raster[~raster_mask]
    unique, counts = np.unique(valid, return_counts=True)

    count_df = pd.DataFrame({
        "Value": unique,
        "Count": counts
    })

    rat_df = pd.merge(value_map, count_df, on="Value", how="left")
    rat_df["Count"] = rat_df["Count"].fillna(0).astype(int)

    if "FID" not in rat_df.columns:
        rat_df.insert(0, "FID", range(len(rat_df)))
    cols = ["FID", "Value", "Count"]

    if field is not None and field in rat_df.columns:
        cols.append(field)
    
    rat_df = rat_df[cols]


    # ---------------------------
    # 10. 保存 RAT
    # ---------------------------
    base = os.path.splitext(out_dest)[0]

    if rat_type.lower() == "csv":
        rat_df.to_csv(base + ".csv", index=False)

    elif rat_type.lower() == "dbf":
        try:
            from dbf import Table, READ_WRITE

            dbf_path = base + ".tif.vat.dbf"

            fields = []
            for col_name, dt in zip(rat_df.columns, rat_df.dtypes):
                if col_name in ["FID", "Value", "Count"]:
                    fields.append(f"{col_name} N(18,0)")
                else:
                    fields.append(f"{col_name} C(254)")

            

            table = Table(dbf_path, ";".join(fields))
            table.open(mode=READ_WRITE)

            for _, row in rat_df.iterrows():
                table.append(tuple(row))

            table.close()

        except ImportError:
            print("dbf 未安装，已自动改为 CSV")
            rat_df.to_csv(base + ".csv", index=False)

    else:
        rat_df.to_csv(base + ".csv", index=False)

    return rat_df



def feature_unify_raster(
    source: Union[str, gpd.GeoDataFrame],
    dst_in: Union[str, rasterio.DatasetReader],
    out_dest: str,
    field: Optional[str] = None,
    dtype=None,
    nodata=None,
    rat: bool = True,
    rat_type: str = 'dbf',
    ):
    
    dataset_opener = get_dataset_opener(dst_in)
    
    with dataset_opener(dst_in) as dst:
        pixel = dst.res[0]
        crs = dst.crs
        bounds = dst.bounds
    feature_to_raster(source, out_dest, pixel,
                      field=field,
                      crs=crs,bounds=bounds, dtype=dtype, nodata=nodata, 
                      rat=rat, rat_type=rat_type)
    
    ...


# ------------------------------------------------------------------ #
# 示例调用
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import tempfile, os
    # from shapely.geometry import box

    gdf_test = gpd.GeoDataFrame(
        {
            "category": ["forest", "water", "urban", "water"],  # 非数值
            "value":    [10.0, 20.0, 30.0, 20.0],              # float
            "count":    [1, 2, 3, 2],                           # int
            "geometry": [box(0,0,1,1), box(1,0,2,1), box(2,0,3,1), box(3,0,4,1)],
        },
        crs="EPSG:4326",
    )

    with tempfile.TemporaryDirectory(dir=r"F:\Python\test") as tmpdir:

        # 1. field=None → int32 burn，rat=True → 生成RAT
        r1 = feature_to_raster(
            source=gdf_test, out_dest=os.path.join(tmpdir, "fid.tif"),
            pixel=0.1, field=None,
        )
        print(f"\nRAT (field=None, dtype自动=int32):\n{r1}\n")

        # 2. 数值float列 → dtype自动=float64，rat=True但浮点跳过
        r2 = feature_to_raster(
            source=gdf_test, out_dest=os.path.join(tmpdir, "value.tif"),
            pixel=0.1, field="value",
        )
        print(f"返回值 (float字段, rat跳过): {r2}\n")

        # 3. 数值int列 → dtype自动=int64，rat=True → 生成RAT
        r3 = feature_to_raster(
            source=gdf_test, out_dest=os.path.join(tmpdir, "count.tif"),
            pixel=0.1, field="count",
        )
        print(f"RAT (int字段, dtype自动=int64):\n{r3}\n")

        # 4. 非数值列 → dtype自动=int32，生成RAT含原始标签
        r4 = feature_to_raster(
            source=gdf_test, out_dest=os.path.join(tmpdir, "category.tif"),
            pixel=0.1, field="category",
        )
        print(f"RAT (非数值列, 含category标签):\n{r4}\n")

        # 5. 强制指定 dtype/nodata，rat=False
        r5 = feature_to_raster(
            source=gdf_test, out_dest=os.path.join(tmpdir, "custom.tif"),
            pixel=0.1, field="value", dtype="float32", nodata=-9999, rat=False,
        )
        print(f"返回值 (rat=False): {r5}")
