# -*- coding: utf-8 -*-

from rastertool.functions import get_dataset_opener, out
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import Affine
import numpy as np
import cmath
import warnings
from rastertool.errors import NodataOverflow
import geopandas as gpd
import xarray as xr
from rasterio.transform import from_origin

from rasterio.crs import CRS


def eq_crs(crs1, crs2):
    return CRS.from_user_input(crs1) == CRS.from_user_input(crs2)

def check_nodata_inrange(nodata, dt):
    '''检测无效值是否在类型的范围内'''
    inrange = False

    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        inrange = info.min <= nodata <= info.max
    else:
        if cmath.isfinite(nodata):
            info = np.finfo(dt)
            inrange = info.min <= nodata <= info.max
            nodata_dt = np.min_scalar_type(nodata)
            inrange = inrange & np.can_cast(nodata_dt, dt)
        else:
            inrange = True
    
    return inrange




def cast_value(x, dtype):
    dt = np.dtype(dtype)      # 统一解析字符串 / 类型 / numpy dtype
    return np.array(x, dtype=dt).item()


def set_nodata(nodataval, dt, nodata=None, dtype=None):
    '''无效值的规范设置'''
    ##nodataval=None, nodataval, dt e
    
    
    src_nodata = nodataval
    src_dt = dt
    
    if dtype is not None:
        dt = dtype
    
    if nodata is not None:
        nodataval = nodata
    
    if nodataval is not None:
        
        if not check_nodata_inrange(nodataval, dt):  # 出现溢出
        
        
            if check_nodata_inrange(nodataval, src_dt):  # nodata可存入源dtype; 回退dtype
                warnings.warn(
                        f'忽略dtype参数。无效值{nodataval}，'
                        f'不能安全的被转换为数据类型{dt}.'
                        '可以使用--dtype 选项来覆盖它，以获得更好的效果。'
                        '恢复使用源数据的数据类型')
                dt = src_dt
            
            elif src_nodata is None:  # 源nodata为None，设置为0
                warnings.warn('nodata is None, set to 0(GDAL default)')
                nodataval = 0
                
            elif check_nodata_inrange(src_nodata, dt):  # 源nodata可存入dtype; 回退nodata
                warnings.warn(
                        f'忽略nodata参数。无效值{nodataval}，不能安全的被转换为数据类型{dt}。'
                        '可以使用 --nodata 选项来覆盖它，以获得更好的效果。恢复使用源数据的无效值'
                        )
                nodataval = src_nodata
            else:  # 源nodata在dtype溢出; 回退nodata、dtype
                if not check_nodata_inrange(src_nodata, src_dt):
                    raise ValueError('源数据无效值与数据类型不匹配')
                warnings.warn(
                        f'忽略nodata与dtype参数。无效值{nodataval}与源数据无效值{src_nodata}，'
                        f'皆不能安全的被转换为数据类型{dt}.'
                        '可以使用 --nodata与--dtype 选项来覆盖它，以获得更好的效果。'
                        '恢复使用源数据的无效值与数据类型')
                dt = src_dt
                nodataval = src_nodata
                
        
    else:
        warnings.warn('nodata is None, set to 0(GDAL default)')
        nodataval = 0
    
    nodataval = cast_value(nodataval, dt)
    
    return nodataval, dt



def copy_raster(source, out_path,
                nodata=None, dtype=None, compress=None,
                update_stats=False,
                **profile_update):
    '''拷贝栅格，可计算统计量与更新元数据'''
    dataset_opener = get_dataset_opener(source)
    
    with dataset_opener(source) as src:
        data = src.read(masked=True)
        mask = data.mask
        
        dt = src.dtypes[0]
        shape = data.shape
        nodataval = src.nodatavals[0]
        
        nodataval, dt = set_nodata(nodataval, dt, nodata, dtype)
        
        
        profile = src.profile
        compress = profile.get('compress', None) if compress is None else compress
        profile.update({'nodata': nodataval, 'dtype': dt, 'compress':compress})
        profile.update(profile_update)
        
    
    dest = np.empty(shape, dtype=dt)
    np.copyto(dest, data, where=~mask)
    np.copyto(dest, nodataval, where=mask)
    
    out(out_path, dest, profile, update_stats=update_stats)



def build_overviews(source, level=4, how=Resampling.nearest):
    '''构建栅格金字塔'''
    factors = [2**(i+1) for i in range(int(level))]
    
    with rasterio.open(source, 'r+') as dataset:
        # 使用最近邻重采样方法构建概视图
        dataset.build_overviews(factors, how)
        # 设置概视图的压缩选项（可选）
        dataset.update_tags(ns='rio_overview', compress='lzw')



def compress_raster(source, out_path, compress='lzw', **kwds):
    '''压缩栅格'''
    copy_raster(source, out_path, compress='lzw', **kwds)




def renan(source, out_path, nodata, dtype=None, compress=None, update_stats=False, **profile_update):
    '''替换无效值'''
    
    dataset_opener = get_dataset_opener(source)
    
    with dataset_opener(source) as src:
        
        data = src.read(masked=True)
        mask = data.mask
        profile = src.profile
        shape = data.shape
        dt = src.dtypes[0]
        
        
        
        if dtype is not None:
            dt = dtype
        
        ### Set nodataval to 0
        if nodata is None:
            nodata = 0
        
        if not check_nodata_inrange(nodata, dt):
            raise NodataOverflow(f"The nodata value, {nodata}, cannot safely be represented in the chosen data type, {dt}.\
                                 Consider modifying 'nodata' or 'dtype'")
        else:
            nodata = cast_value(nodata, dt)
        
        dest = np.empty(shape, dtype=dt)
        np.copyto(dest, data, where=~mask)
        np.copyto(dest, nodata, where=mask)
        
        compress = profile.get('compress', None) if compress is None else compress
        
        profile.update({'dtype': dt, 'nodata': nodata, 'compress':compress})
        profile.update(profile_update)
        
    
    out(out_path, dest, profile, update_stats=update_stats)





def polygon_to_raster(shp,raster,pixel,field,
                      crs=None,dtype='float32',nodata=-9999):
    '''
    矢量转栅格
    :param shp: 输入矢量全路径，字符串，无默认值
    :param raster: 输出栅格全路径，字符串，无默认值
    :param pixel: 像元大小，与矢量坐标系相关
    :param field: 栅格像元值字段
    :param crs: 输出坐标系代码，默认为4326
    :return: None
    '''

    # 判断字段是否存在
    if crs:
        shapefile = gpd.read_file(shp).to_crs(crs)
    else:
        shapefile = gpd.read_file(shp)
        crs = shapefile.crs
    if not field in shapefile.columns:
        raise Exception ('输出字段不存在')
    shapefile[field] = shapefile[field].astype(dtype)

    bound = shapefile.bounds
    width = int((bound.get('maxx').max()-bound.get('minx').min())/pixel)
    height = int((bound.get('maxy').max()-bound.get('miny').min())/pixel)
    transform = Affine(pixel, 0.0, bound.get('minx').min(),
           0.0, -pixel, bound.get('maxy').max())

    meta = {'driver': 'GTiff',
            'dtype': dtype,
            'nodata': nodata,
            'width': width,
            'height': height,
            'count': 1,
            'crs': crs,
            'transform': transform}

    with rasterio.open(raster, 'w+', **meta) as out:
        out_arr = out.read(1)
        shapes = ((geom,value) for geom, value in zip(shapefile.get('geometry'), shapefile.get(field)))
        burned = features.rasterize(shapes=shapes, fill=nodata, out=out_arr, transform=out.transform)
        out.write_band(1, burned)
        out.statistics(1, clear_cache=True)



def nc_to_raster(ds_nc,
              vr=None,
              out_ph=None,
              loc_names=None,
              crs='EPSG:4326',
              dtype='float32',
              nodata=None,
              desc:list[str] | str=None):
    '''
    nc转tif
    仅限维度为二[lat,lon] --- [height, width]
    或维度大小为三[*, lat,lon] --- [count, height, width] 第三个维度将设为栅格波段维度
    *维度顺序不限将自动排序

    Parameters
    ----------
    ds_nc : xr.Dataset,xr.DataArray
        nc变量
    vr : str
        提取的变量, 当vr为None且ds_nc为xr.Dataset时，ds_nc=ds_nc.to_dstaarray()
    out_ph : str
        输出位置
    loc_names : dict, optional
        lon,lat的对应变量名,默认为lon,lat
        
        eg. {'lon':'longitude', 'lat':'latitude'}
        
    crs : str, dict, or CRS; optional
        空间参考. The default is 'EPSG:4326'.
    dtype :  str or numpy dtype, optional
        数据类型. The default is 'float32'.
    nodata : int, float, or nan; optional
        数据无效值.
    desc : list[str] | str ,optional
        波段名列表，维度为3时默认为波段维度data, 维度为2时默认为 None

    Returns
    -------
    if out_ph is None :
        return arr_vr, profile, descriptions(波段名)
    else:
        output tif and return out_ph

    '''
    
   
    if loc_names is None:
        loc_names = {}
    
    lon_name = loc_names.get('lon', 'lon')
    lat_name = loc_names.get('lat', 'lat')
    
    if not isinstance(ds_nc, xr.DataArray):
        if vr is None:
            ds_nc = ds_nc.to_dataarray()
        else:
            ds_nc = ds_nc[vr].dims
    else:
        pass
    
    # dims判断
    dims = list(ds_nc.dims)
    assert {lat_name,lon_name}.issubset(set(dims)) , '未找到代表lat、lon的维度，尝试重新定义loc_names参数'
    assert len(dims) <= 3, 'dims长度超限,期望长度2或3，得到%d。仅接受代表[lat, lon]或[band, lat, lon]维度组'%len(dims)
        
    # 维度排序, descriptions获取
    loc = [lat_name, lon_name]
    if len(dims) == 3:
        new_dims = [i for i in dims if i not in loc] + loc
        descriptions = (ds_nc[new_dims[0]].data.astype(str).tolist() if desc is None
                        else
                        desc)
        count = ds_nc[new_dims[0]].size
    else:
        new_dims = loc
        descriptions = ([None] if descriptions is None
                        else
                        descriptions)
        count = 1
    
    descriptions = ([descriptions] if isinstance(descriptions, str)
                    else
                    descriptions)
    assert len(descriptions) == count, 'descriptions长度%d与波段数%d不一致' % (len(descriptions), count)
    
    if dims != new_dims:
        ds_nc = ds_nc.transpose(*new_dims)
    
    
    # 获取经纬度序列
    lon = ds_nc[lon_name].data
    lat = ds_nc[lat_name].data
    
    # 获取分辨率
    res_lon = abs(lon[1] - lon[0])
    res_lat = abs(lat[1] - lat[0])
    
    # 计算栅格transform
    transform = from_origin(west=lon.min()-res_lon/2,
                            north=lat.max()+res_lat/2,
                            xsize=res_lon, ysize=res_lat)
    
    
    # 获取数据矩阵
    arr_vr = ds_nc.data

    
    # 统一为三维(count, height, width), 并获取shape
    if arr_vr.ndim == 2:
        arr_vr = np.array([arr_vr])
    count, height, width = arr_vr.shape
    
    
    # 检查经纬度方向, 保证lat递减, lon递增
    if lat[1] > lat[0]:
        arr_vr = np.flip(arr_vr,axis=1) 
    if lon[1] < lon[0]:
        arr_vr = np.flip(arr_vr,axis=2) 
    
    # 定义profile
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'lzw',
        'interleave': 'band',
    }
    
    if out_ph is None:
        return arr_vr, profile, descriptions
    else:
        # 输出
        with rasterio.open(out_ph, 'w', **profile) as dst:
            dst.write(arr_vr)
            
            dst.update_stats()  # raserio >= 1.4.0
            dst.descriptions = descriptions
            
        return out_ph


def get_geometry(ph_shp, crs=None):
    shp = gpd.read_file(ph_shp) if crs is None else gpd.read_file(ph_shp).to_crs(crs)
    return shp.geometry



def rio_mask(dataset, shapes,
             nodata=None,dtype=None,
             crop=True,all_touched=False,filled=True,
             **kwgs):
    """
    使用矢量形状对栅格数据进行掩膜操作，并返回掩膜后的数组和更新后的元信息。

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        将应用掩膜的栅格数据集
    shapes : iterable
        矢量形状列表，用于掩膜操作。
    crop : bool, optional
        是否裁剪到矢量范围，默认为 True。
    nodata : int or float, optional
        无效值。如果未提供，则使用输入栅格的 nodata 值；如果栅格没有 nodata 值，则默认为 0。
    dtype : data-type, optional
        默认情况下，数据类型是从输入数据推断出来的
    all_touched : bool, optional
        如果为 True，则掩膜操作中会填充所有被形状边界触碰的像素；否则只填充形状内部的像素。
    filled : bool, optional
        如果为 True，则掩膜外填充为 nodata；否则返回 fill_value 为 nodata 的掩膜数组。
        nodata 无法转换为 dataset 的 dtype 时可设置 dtype 参数，否则会报错。
    **kwgs : dict
        其他关键字参数，传递给 `rasterio.mask.mask`。

    Returns
    -------
    arr : numpy.ndarray
        掩膜处理后的数组。
    profile : dict
        更新后的数据集元信息（profile）。
    """
    # crs = src_in.crs
    
    # shp = gpd.read_file(ph_shp).to_crs(crs)
    # shapes = shp.geometry
    
    # `profile`获取
    profile = dataset.profile.copy()
    dt = dataset.dtypes[0]
    nodataval = dataset.nodatavals[0]
    
    # `nodata`获取
    nodata, dtype = set_nodata(nodataval, dt, nodata, dtype)
    
    
    
    # 运行`rasterio.mask.mask`, 关闭filled(改为由后续代码自定义控制)
    arr, tf = rasterio.mask.mask(dataset, shapes,
                                 crop=crop,nodata=nodata,
                                 all_touched=all_touched,
                                 filled=False,
                                 **kwgs)
    
    # 更新`arr`的类型
    # if dtype is not None:
    arr = np.ma.array(arr.data.astype(dtype), mask=arr.mask)
    
    # 检查无效值能否储存在`arr`中
    # dtype = arr.dtype
    try:
        fill_value = np.asarray(nodata, dtype=dtype)
    except (OverflowError, ValueError) as e:
        # Raise TypeError instead of OverflowError or ValueError.
        # OverflowError is seldom used, and the real problem here is
        # that the passed fill_value is not compatible with the ndtype.
        err_msg = "Cannot convert nodata %s to dtype %s"
        raise TypeError(err_msg % (fill_value, dtype)) from e
    
    # 填充设置
    if filled:
        arr = arr.filled(nodata)
    else:
        arr.fill_value = nodata
    
    
    # 更新`profile`
    profile.update({
                'dtype': arr.dtype,
                'nodata':nodata,
                "height": arr.shape[1],
                "width": arr.shape[2],
                "transform": tf})
    return arr, profile




def mask_shp(source, mask, out_path=None,
             nodata=None,dtype=None,
             crop=True,all_touched=False,filled=True,
             **kwgs):
    
    dataset_opener = get_dataset_opener(source)
    
    
    with dataset_opener(source) as src:
        crs = src.crs
        
        shapes = gpd.read_file(mask)
        if not eq_crs(shapes.crs, crs):
            shapes = shapes.to_crs(src.crs)
        
    
        dest, profile = rio_mask(dataset=src, shapes=shapes.geometry.values,
                                 nodata=nodata,dtype=dtype,
                                 crop=crop,all_touched=all_touched,filled=filled,
                                 **kwgs)
    
    if out_path is None:
        return dest, profile
    else:
        out(out_path, dest, profile)
























