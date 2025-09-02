#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:08:02 2023

@author: vpremier
"""

import json
import os
from datetime import datetime
from osgeo import gdal, osr, ogr
import numpy as np
import netCDF4


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config



def check_config_consistency(config):
    """
    Validate required config fields for preprocessing.
    Raises ValueError if any check fails.
    """
    # Flags: check they exist and are bool
    for flag in ["query_landsat", "query_sentinel2", "download_landsat", "download_sentinel2"]:
        if flag not in config:
            raise ValueError(f"Missing required flag: '{flag}'")
        if not isinstance(config[flag], bool):
            raise ValueError(f"Flag '{flag}' must be a boolean.")

    # Output directory: must be string and not empty
    outdir = config.get("output_directory")
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError("'output_directory' must be a non-empty string.")

    # Shapefile: must be string and not empty
    shp = config.get("shapefile")
    if not isinstance(shp, str) or not shp.strip():
        raise ValueError("'shapefile' must be a non-empty string.")
    if not os.path.isfile(shp):
        raise ValueError(f"Shapefile path does not exist: '{shp}'")

    # Dates: must be string and valid dates
    date_start = config.get("date_start")
    date_end = config.get("date_end")
    if not isinstance(date_start, str) or not isinstance(date_end, str):
        raise ValueError("'date_start' and 'date_end' must be strings in 'YYYY-MM-DD' format.")
    try:
        start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")
    if start_dt >= end_dt:
        raise ValueError("'date_start' must be before 'date_end'.")

    # Cloud cover: int in [0, 100]
    max_cc = config.get("max_cloudcover")
    if not isinstance(max_cc, int) or not (0 <= max_cc <= 100):
        raise ValueError("'max_cloudcover' must be an integer between 0 and 100.")

    # Landsat satellite list: must be list with valid options
    landsat_satellite = config.get("landsat_satellite")
    valid_satellites = {"LT05", "LE07", "LC08", "LC09"}
    if not isinstance(landsat_satellite, list):
        raise ValueError("'landsat_satellite' must be a list.")
    invalid = [s for s in landsat_satellite if s not in valid_satellites]
    if invalid:
        raise ValueError(f"Invalid Landsat satellites: {invalid}. Allowed: {valid_satellites}.")

    # s2_tile_list and landsat_tile_list: must be lists (can be empty)
    for tile_key in ["s2_tile_list", "landsat_tile_list"]:
        if tile_key not in config:
            raise ValueError(f"Missing '{tile_key}' in config.")
        if not isinstance(config[tile_key], list):
            raise ValueError(f"'{tile_key}' must be a list.")




def reproj_point(x, y, srIn, srOut):
    """Reproject a point into a defined coordinate system.
    
    Parameters
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    srIn : osgeo.osr.SpatialReference
        spatial reference of the input coordinate system
    srOut : osgeo.osr.SpatialReference
        spatial reference of the output coordinate system
        
    Returns
    -------
    (x, y) : tuple
        the transformed coordinates
    """
    
    epsgList = ['4326', '31287']
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    
    if int(gdal.__version__[0]) >= 3 and srIn.GetAttrValue("AUTHORITY", 1) in epsgList:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        point.AddPoint(y, x)
    else:
        point.AddPoint(x, y)
    
    coordTransform = osr.CoordinateTransformation(srIn,srOut)
    
    # transform point
    point.Transform(coordTransform)
    
    if int(gdal.__version__[0]) >= 3 and srOut.GetAttrValue("AUTHORITY", 1) in epsgList:
        (y, x) = point.GetX(), point.GetY()
    else:
        (x, y) = point.GetX(), point.GetY()
        
    return (x,y)




def get_closest_extent(info, epsg_target, res, prec =5):
    """Given two images with two different crs, gets the array indices (of the 
    top left and right bottom corners) of the second (target) image in the first 
    (source) image and the new geotransform (in the source crs).
    
    Parameters
    ----------
    info_target : dict
        dictionary containing the target image metadata (see the function open_image)
        
    info_src : dict
        dictionary containing the source image metadata    
        
    buffer : int
        number of pixels in the buffer area
    
    Returns
    -------
    x_tl : float
        the x-coordinate of the scene's top left corner point
    y_br : float
        the y-coordinate of the scene's bottom right corner point
    x_br : float
        the x-coordinate of the scene's bottom right corner point
    y_tl : float
        the y-coordinate of the scene's top left corner point
    geotransform : tuple
        the geotransform information used to georeference the image
    """
    
    srOut = osr.SpatialReference()
    srOut.ImportFromEPSG(int(epsg_target))
    
    # reference points to project: top left, bottom right
    xtl = info['geotransform'][0]
    ytl = info['geotransform'][3]
    xbr = (info['geotransform'][0] +
              info['X_Y_raster_size'][0] * 
              info['geotransform'][1])
    ybr = (info['geotransform'][3] +
              info['X_Y_raster_size'][1] *
              info['geotransform'][5])
    

    srIn = osr.SpatialReference()
    srIn.ImportFromWkt(info['projection'])
    
    # reproject reference points
    (xtl_r, ytl_r) = reproj_point(xtl, ytl, srIn, srOut)
    (xtr_r, ytr_r) = reproj_point(xbr, ytl, srIn, srOut)
    (xbr_r, ybr_r) = reproj_point(xbr, ybr, srIn, srOut)
    (xbl_r, ybl_r) = reproj_point(xtl, ybr, srIn, srOut)
    
    # extent of the image
    xMin = min(xtl_r, xbl_r)
    xMax = max(xtr_r, xbr_r)
    yMin = min(ybr_r, ybl_r)
    yMax = max(ytl_r, ytr_r)
    
    
    # find extent in the new reference system
    xMin = round(int(xMin / res) * res, prec)
    yMin = round(int(yMin / res) * res, prec)
    xMax = round(np.ceil(xMax / res) * res, prec)
    yMax = round(np.ceil(yMax / res) * res, prec)
    
    # check if the extent exceeds 180 degrees longitude
    # if yes, correct reset to -180 degrees by adding 360 degrees
    if xMax < xMin and epsg_target == '4326':
        xMax = xMax + 360
        
    return xMin, yMin, xMax, yMax

   

    

def open_image(image_path):
    """Opens an image and reads its metadata.
    
    Parameters
    ----------
    image_path : str
        path to an image
    
    Returns
    -------
    image : osgeo.gdal.Dataset
        the opened image
    information : dict
        dictionary containing image metadata    
    """
    
    ext = os.path.basename(image_path).split('.')[-1]
    
    if ext == 'nc':
        nc_data = netCDF4.Dataset(image_path,'r')
        vars_nc = list(nc_data.variables)
        scf_name = list(filter(lambda x: x.startswith('scf'), vars_nc))[0]
        
        image = gdal.Open("NETCDF:{0}:{1}".format(image_path, scf_name))

            

    else:
        image = gdal.Open(image_path)
    
    if image is None:
        print('could not open ' + image_path)
        return
        
    cols = image.RasterXSize
    rows = image.RasterYSize
    geotransform = image.GetGeoTransform()
    proj = image.GetProjection()
    minx = geotransform[0]
    maxy = geotransform[3]
    maxx = minx + geotransform[1] * cols
    miny = maxy + geotransform[5] * rows
    X_Y_raster_size = [cols, rows]
    extent = [minx, miny, maxx, maxy]
    information = {}
    information['geotransform'] = geotransform
    information['extent'] = extent
    information['X_Y_raster_size'] = X_Y_raster_size
    information['projection'] = proj

    if ext == 'nc':
        information['geotransform'] = tuple(map(lambda x: round(x, 2) or x, information['geotransform']))
        information['extent'] = tuple(map(lambda x: round(x, 2) or x, information['extent']))

    return image, information








          
            

  
