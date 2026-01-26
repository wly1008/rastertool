# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:03:00 2025

@author: wly
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd

from pyproj import CRS


# import mycode.data.chinaShp.GS_2024_0650 as data


dir_shp = r'D:/app/anaconda3/envs/py312/Lib/mycode/data/chinaShp/GS_2024_0650/CGCS2000'

## CGCS2000
out_dir = None
out_dir_aea = r'D:/app/anaconda3/envs/py312/Lib/mycode/data/chinaShp/GS_2024_0650/CGCS2000/Albers'
crs = None
crs_aea = CRS.from_string('PROJCS["CGCS2000_Albers_China",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",105.0],PARAMETER["Standard_Parallel_1",25.0],PARAMETER["Standard_Parallel_2",47.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')



# WGS1984
# out_dir = r'D:/app/anaconda3/envs/py312/Lib/mycode/data/chinaShp/GS_2024_0650/WGS84'
# out_dir_aea = r'D:/app/anaconda3/envs/py312/Lib/mycode/data/chinaShp/GS_2024_0650/WGS84/Albers'

# crs = CRS.from_string('+proj=longlat +datum=WGS84 +no_defs')
# crs_aea = CRS.from_string('PROJCS["WGS84_Albers_China",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",105.0],PARAMETER["Standard_Parallel_1",25.0],PARAMETER["Standard_Parallel_2",47.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')



# # Krasovsky_1940
# out_dir = r'D:/app/anaconda3/envs/py312/Lib/mycode/data/chinaShp/GS_2024_0650/KRASS1940'
# out_dir_aea = r'D:/app/anaconda3/envs/py312/Lib/mycode/data/chinaShp/GS_2024_0650/KRASS1940/Albers'

# crs = CRS.from_epsg(4024)
# crs_aea = CRS.from_string('PROJCS["Krasovsky_1940_Albers_China",GEOGCS["GCS_Krasovsky_1940",DATUM["D_Krasovsky_1940",SPHEROID["Krasovsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",105.0],PARAMETER["Standard_Parallel_1",25.0],PARAMETER["Standard_Parallel_2",47.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')




file_shps = [i for i in os.listdir(dir_shp) if i.endswith('.shp')]





for file in file_shps:
    
    ph_shp = os.path.join(dir_shp, file)
    
    
    out_ph_aea = os.path.join(out_dir_aea, file)
    
    gpd.read_file(ph_shp).to_crs(crs_aea).to_file(out_ph_aea)
    if out_dir is None:
        continue
    out_ph = os.path.join(out_dir, file)
    gpd.read_file(ph_shp).to_crs(crs).to_file(out_ph)
    

























