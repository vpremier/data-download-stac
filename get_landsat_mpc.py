#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:52:10 2024

@author: vpremier
"""
import os
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import CRS
import numpy as np
from rasterio.transform import Affine
from osgeo import gdal, osr
import odc.stac
import planetary_computer
import pystac_client
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from utils import get_closest_extent, open_image

import logging


def define_info(info_src, extent_target = None, epsg_target = None, 
                resolution = None, img4ext = None):
    """
    Get new information dictionary. Extent, epsg and resolution
    can be either user defined or the raw ones (of the original image), if 
    no parameter is specified. 

    Parameters
    ----------
    info_src : dict
        dictionary with the information of the source image.
    extent_target : tuple, optional
        (xmin, ymin, xmax, ymax). The default is None.
    epsg_target : str, optional
        EPSG of the target reference system. The default is None.
    resolution : int, optional
        Target resolution. The default is None.
    img4ext : str, optional
        Path of an image to bet used to set the extent/epsg. The default is False.


    Returns
    -------
    info_target : dict
        Target informations (extent, resolution, epsg) stored as dictionary.

    """
    
    if extent_target is None and img4ext is None:
        
        # keep the original extent. It is possible to define the target crs
        # and/or the target resolution
        info_target = info_src.copy()
        print('Keeping the extent of the source image..')

        if epsg_target is not None and resolution is not None:
            print('Setting EPSG:%s and resolution %i m' %(epsg_target, resolution))
            
            # reference system
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(epsg_target))
            
            
            xMin, yMin, xMax, yMax = get_closest_extent(info_src, epsg_target, 
                                                        resolution, prec =2)
            
            x_size = int(np.round((xMax - xMin)/resolution))
            y_size = int(np.round((yMax - yMin)/resolution))
        
            info_target = {'geotransform':(xMin, resolution, 0, 
                                            yMax, 0, -resolution),
                            'extent': [xMin, yMin, xMax, yMax],
                            'X_Y_raster_size': [x_size, y_size],
                            'projection': srs.ExportToWkt()}
        
        elif resolution is not None:
            print('Keeping native crs and setting resolution %i m' %resolution)
            
            x_size = int(np.round((info_src['extent'][2] - \
                                   info_src['extent'][0])/resolution))
            y_size = int(np.round((info_src['extent'][3] - \
                                   info_src['extent'][1])/resolution))
        
            info_target['geotransform'] = (info_src['geotransform'][0], resolution,
                                           0, info_src['geotransform'][3], 0, 
                                           -resolution)
            info_target['X_Y_raster_size'] = [x_size, y_size]
            
            
    else:
        
        print('User defined extent, epsg and resolution..')
        # user defined extent and epsg.
            
        # Read the target extent and crs from another image or 
        #from the user specified parameters
        
        if img4ext is not None:
            print('Reading extent and epsg from another image')
            ds, info_target = open_image(img4ext)
            extent_target = info_target['extent']
            
        else:  
            
            assert epsg_target, "Please specify the target EPSG or enter the path to a target image"

            # reference system
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(epsg_target))
            
            info_target = {}
            info_target['projection'] = srs.ExportToWkt()
           
          
        assert resolution, "Please specify the target resolution"
        
        x_rest = (extent_target[2] - extent_target[0])%resolution
        y_rest = (extent_target[3] - extent_target[1])%resolution
            
        #assert (x_rest ==0 and y_rest==0),'The extent is not multiple of the assigned resolution'

        
        # get the number of pixels
        x_size = int(np.round((extent_target[2] - extent_target[0])/resolution))
        y_size = int(np.round((extent_target[3] - extent_target[1])/resolution))
        
        
        # store information in a dictionary
        info_target['geotransform'] = (extent_target[0], resolution, 0, 
                                       extent_target[-1], 0, -resolution)
        info_target['extent'] = list(extent_target)
        info_target['X_Y_raster_size'] = [x_size, y_size]

        
    return info_target



def get_bbox_wgs84(img4ext=None, extent_target=None, epsg_target=None, buffer_m=0):
    """
    Returns bbox [xmin, ymin, xmax, ymax] in WGS84.
    
    Parameters
    ----------
    img4ext : str, optional
        Path to reference raster.
    extent_target : tuple, optional
        Target extent (xmin, ymin, xmax, ymax).
    epsg_target : int or str, optional
        EPSG code of extent_target.
    buffer_m : float, optional
        Buffer in CRS units (e.g., meters if projected).
    """

    if img4ext:
        with rio.open(img4ext) as src:
            bounds = src.bounds
            crs = src.crs
        geom = gpd.GeoSeries([box(*bounds)], crs=crs)

    elif extent_target and epsg_target:
        crs = CRS.from_user_input(f"EPSG:{epsg_target}")
        geom = gpd.GeoSeries([box(*extent_target)], crs=crs)

    else:
        raise ValueError("Must provide either img4ext OR (extent_target + epsg_target)")

    # Apply buffer if requested
    if buffer_m > 0:
        geom = geom.buffer(buffer_m)

    # Convert to WGS84 and extract bbox
    geom_wgs84 = geom.to_crs(4326)
    return geom_wgs84.total_bounds.tolist()



def convert_landsat_bands(outdir, date_start, date_end, resolution=None, 
                          img4ext = None, extent_target=None, epsg_target=None,
                          reproj_type=Resampling.bilinear, suffix='_boa',
                          na_value = "NaN", calibration=True, ow=False):        
    """
    Converts and optionally reprojects Landsat bands from an xarray dataset to single-band GeoTIFFs.

    This function supports reprojection based on user-defined resolution, extent, EPSG code,
    or by matching the spatial configuration of a reference image (`img4ext`). Bands are 
    saved individually as GeoTIFF files.

    Parameters:
        data (xarray.Dataset): Dataset loaded using odc.stac.stac_load containing Landsat bands.
        outdir (str): Output directory where GeoTIFFs will be saved.
        image_id (str): Identifier for the Landsat scene, used to name output files.
        bands (dict): Dictionary mapping xarray band names to short names (e.g., {"SR_B2": "blue"}).
        resolution (float, optional): Target spatial resolution in meters. If None, native resolution is used.
        img4ext (str, optional): Path to a reference image whose extent and projection will be used.
        extent_target (list, optional): Custom output extent [xmin, ymin, xmax, ymax]. Ignored if `img4ext` is set.
        epsg_target (int or str, optional): Target coordinate reference system (EPSG code). Required if reprojection is desired.
        reproj_type (rasterio.enums.Resampling, optional): Resampling method for reprojection. Default is bilinear.
        suffix (str, optional): Suffix to add to the output filenames (e.g., "_toa" or "_boa").
        na_value (str or float, optional): Value used to represent NoData in the output. Default is "NaN".
        calibration (bool, optional): Whether to apply reflectance calibration to the bands. Default is True.
        ow (bool, optional): Overwrite existing output files. Default is False.

    Returns:
        None. Saves one GeoTIFF file per band in the specified output directory.
    """
    
    # determine AOI bbox
    bbox_of_interest = get_bbox_wgs84(img4ext=img4ext, 
                                      extent_target=extent_target, 
                                      epsg_target=epsg_target, 
                                      buffer_m=1000)
    
    
    # dictionary with the bands   
    bands_dic = {"LT05" : {"blue" : "SR_B1", 
                            "green" : "SR_B2",
                            "red" : "SR_B3",
                            "nir08" : "SR_B4", 
                            "swir16" : "SR_B5",  
                            "trad" : "ST_B6",
                            "swir22" : "SR_B7"},
                 "LE07" :  {"blue" : "SR_B1", 
                            "green" : "SR_B2",
                            "red" : "SR_B3",
                            "nir08" : "SR_B4", 
                            "swir16" : "SR_B5",  
                            "trad" : "ST_B6",
                            "swir22" : "SR_B7"},
                 "LC08" :  {"coastal" : "SR_B1",
                            "blue" : "SR_B2", 
                            "green" : "SR_B3",
                            "red" : "SR_B4",
                            "nir08" : "SR_B5", 
                            "swir16" : "ST_B6",  
                            "swir22" : "SR_B7",
                            "trad" : "ST_B10"},
                 "LC09" :  {"coastal" : "SR_B1",
                            "blue" : "SR_B2", 
                            "green" : "SR_B3",
                            "red" : "SR_B4",
                            "nir08" : "SR_B5", 
                            "swir16" : "SR_B6",  
                            "swir22" : "SR_B7",
                            "trad" : "ST_B10"}
                 }
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    
    start = pd.to_datetime(date_start, format="%Y-%m-%d", errors="raise")
    end = pd.to_datetime(date_end, format="%Y-%m-%d", errors="raise")
    if start >= end:
        raise ValueError("date_start must be earlier than date_end")
    time_of_interest = f"{date_start}/{date_end}"
    
    
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 80}},
    )
    
    items = search.item_collection()
    print(f"Returned {len(items)} Items")

    os.makedirs(outdir, exist_ok=True)

    log_file = os.path.join(outdir, "landsat_conversion.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Started Landsat download from Microsoft Planetary Computer")
    
    for item in items:
        
        # id of the scene 
        image_id = item.id
        print("Processing %s " %image_id)
        logging.info(f"Processing {image_id}")
        
        sensor = item.id.split('_')[0]
        bands = bands_dic[sensor]
        
        # create folder
        os.makedirs(os.path.join(outdir, sensor), exist_ok=True)
        os.makedirs(os.path.join(outdir, sensor, f"{image_id}"), exist_ok=True)


        try:
            # load the bands as xarray dataset
            data = odc.stac.stac_load(
                [item], bands=bands, bbox=bbox_of_interest
            ).isel(time=0)
        except Exception as e:
            msg = f"Failed to load data for {image_id}: {str(e)}"
            logging.error(msg)
            print(msg)
            continue
        
        # read the MTL file
        mtl_url = item.assets["mtl.txt"].href
        mtl_outfile = os.path.join(outdir, sensor,  f"{image_id}", f"{image_id}_MTL.txt")
    
    
        # Download and save
        response = requests.get(mtl_url)
        if response.ok:
            with open(mtl_outfile, "w") as f:
                f.write(response.text)
            print(f"Saved MTL file to: {mtl_outfile}")
        else:
            print(f"Failed to download MTL from: {mtl_url}")
            
 
    
        # === Extract info_src from xarray ===
        gt_str = data['spatial_ref'].attrs['GeoTransform']
        gt_vals = list(map(float, gt_str.split(' ')))
        transform = Affine(gt_vals[1], gt_vals[2], gt_vals[0],
                           gt_vals[4], gt_vals[5], gt_vals[3])
      
        width = data.dims['x']
        height = data.dims['y']
        extent_src = [
            gt_vals[0],                               # xmin
            gt_vals[3] + gt_vals[5] * height,         # ymin
            gt_vals[0] + gt_vals[1] * width,          # xmax
            gt_vals[3]                                # ymax
        ]
        
        
        info_src = {
            'geotransform': transform.to_gdal(),
            'extent': extent_src,
            'X_Y_raster_size': [width, height],
            'projection': data['spatial_ref'].attrs['crs_wkt']
        }
          
        
        # === Compute target info ===
        info_target = define_info(info_src, extent_target=extent_target,
                                  epsg_target=epsg_target, resolution=resolution,
                                  img4ext=img4ext)
    
    
        dst_transform = Affine(info_target['geotransform'][1], 
                               info_target['geotransform'][2], 
                               info_target['geotransform'][0],
                               info_target['geotransform'][4], 
                               info_target['geotransform'][5], 
                               info_target['geotransform'][3])
        dst_crs = CRS.from_wkt(info_target['projection'])
        dst_width, dst_height  = info_target['X_Y_raster_size']
        
    
        # Iterate through bands
        for band_name in data.data_vars:
          
            out_path = os.path.join(outdir, sensor, f"{image_id}", f"{image_id}_{bands[band_name]}_boa.tif")
            if os.path.exists(out_path) and not ow:
                print(f"Skipping {out_path} (already exists)")
                continue
            
            
            band_data = data[band_name].values.astype("float32")
            nodata_val = data[band_name].attrs.get('nodata', -9999)
    
            band_data[band_data == nodata_val] = np.nan
            
     
            # === Apply constants  ===
            if calibration and band_name != "trad":
                # Landsat Collection 2 L2 scaling
                # see https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1618_Landsat-4-7_C2-L2-ScienceProductGuide-v4.pdf
                # see https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v6.pdf
                band_data = band_data * 0.0000275 - 0.2 
            elif calibration and band_name == "trad":
                band_data = band_data * 0.00341802 + 149
    
    
            # === Reproject ===
            reprojected = np.empty((dst_height, dst_width), dtype='float32')
            reproject(
                source=band_data,
                destination=reprojected,
                src_transform=transform,
                src_crs=CRS.from_wkt(info_src['projection']),
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=reproj_type
            )
    
       
    
            # === Save GeoTIFF ===
            profile = {
                'driver': 'GTiff',
                'height': dst_height,
                'width': dst_width,
                'count': 1,
                'dtype': 'float32',
                'crs': dst_crs,
                'transform': dst_transform,
                'nodata': np.nan,
            }
            
    
 
    
            with rio.open(out_path, 'w', **profile) as dst:
                dst.write(reprojected, 1)
    
            print(f"Saved {out_path}")



if __name__ == "__main__":
    
    # set the directories
    outdir =r'/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Azufre/landsat'
    
    # Base path where all area folders will be created
    base_outdir = r'/mnt/CEPH_PROJECTS/SNOWCOP/Paloma'
    epsg_target = 32719
    
    
    date_start = "2013-04-01"
    date_end = "2023-03-31"
    
    
    # Define all areas with their extents (replace with the correct values for Area1..Area10)
    areas = {
        # "Area01":  (375500, 6819500, 427500, 6878500),
        "Area02":  (375500, 6642500, 427500, 6701500),
        # # "Area03":  (335000, 6560000, 390500, 6624000),
        # "Area04":  (340000, 6417500, 395500, 6478500),
        # #"Area05":  (390000, 6304000, 448500, 6400000),
        # #"Area06":  (366000, 6205000, 428500, 6342500),
        # "Area07":  (342000, 6084000, 398500, 6158500),
        # "Area08":  (323500, 5993500, 375500, 6052500),
        # "Area09":  (288000, 5970500, 328500, 6011000),
        # "Area10": (271500, 5875500, 323500, 5934500),
    }
    
    for area_name, extent_target in areas.items():
        # Construct area-specific output folder
        outdir = os.path.join(base_outdir, area_name, "Landsat")
        os.makedirs(outdir, exist_ok=True)
         
        convert_landsat_bands(outdir, date_start, date_end, resolution=50,
                              img4ext = None, extent_target=extent_target, 
                              epsg_target=epsg_target,
                              reproj_type=Resampling.bilinear, suffix='_boa',
                              na_value = "NaN", calibration=True, ow=False)
    
    
    # to do: filter by RON, add time to inpuit parameter, pulisci la funzione
    # struttura divisa per tile folder
    
    


        
        
        