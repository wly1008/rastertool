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


def feature_to_raster(
    source: Union[str, gpd.GeoDataFrame],
    out_dest: str,
    pixel: float,
    field: Optional[str] = None,
    crs=None,
    dtype=None,
    nodata=None,
    rat: bool = True,
    rat_type: str = 'dbf',
) -> Optional[pd.DataFrame]:

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

    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=dtype
    )

    # ---------------------------
    # 8. 写出 raster
    # ---------------------------
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

    # ---------------------------
    # 9. RAT 生成
    # ---------------------------
    if not rat:
        return None

    if not np.issubdtype(dtype_np, np.integer):
        return None

    # 像元统计
    valid = raster[raster != nodata]
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
# ------------------------------------------------------------------ #
# 示例调用
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import tempfile, os
    from shapely.geometry import box

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
