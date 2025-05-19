from pathlib import Path
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgpd
import ee
import streamlit as st

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

from solmate_app.config import DATA_DIR
from solmate_app.config import COMMON_CRS, COMMON_SCALE

def _read(path): return gpd.read_file(Path(DATA_DIR, path))

@st.cache_data
def load_static_layers():
    residential_1 = _read("residential_areas_part1.geojson")
    residential_2 = _read("residential_areas_part2.geojson")
    main_roads_1 = _read("philippines_main_roads_1.geojson")
    main_roads_2 = _read("philippines_main_roads_2.geojson")

    return {
        "gdf_protected": dgpd.read_parquet(f"{DATA_DIR}/protected_areas_reprojected.parquet").compute(),
        "gdf_kba": _read("philippines_kba.geojson"),
        "gdf_spug":  _read("philippines_spug.geojson"),
        "gdf_airports": _read("philippines_airports.geojson"),
        "faults_geom": _read("faults_ph_geometry.geojson"),
        "gdf_grid": _read("philippines_grid.geojson"),
        "gdf_residential": pd.concat([residential_1, residential_2], axis=0, ignore_index=True),
        "gdf_main_roads": pd.concat([main_roads_1, main_roads_2], axis=0, ignore_index=True),
    }

@st.cache_resource
def load_gee_data():
    # Load SRTM elevation data
    srtm = ee.Image('USGS/SRTMGL1_003')

    # Parameters for Global Flood Database
    start_year = 2000
    end_year = 2018
    n_years = end_year - start_year + 1

    # Flood Data
    flood_collection = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1") \
        .select("flooded") \
        .map(lambda img: img.unmask(0))  # Replace null/masked values with 0
    flood_occurrence_sum = flood_collection.filterDate(f"{start_year}-01-01", f"{end_year}-12-31").sum()
    flood_duration = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1") \
        .select("duration") \
        .map(lambda img: img.unmask(0))  # Replace null/masked values with 0
    flood_duration_sum = flood_duration.filterDate(f"{start_year}-01-01", f"{end_year}-12-31").sum()
    flood_depth_collection = ee.ImageCollection("JRC/CEMS_GLOFAS/FloodHazard/v1") \
        .select("depth") \
        .map(lambda img: img.unmask(0))  # Replace null/masked values with 0
    nonzero_depth = flood_depth_collection.map(lambda img: img.updateMask(img.gt(0))) # include only non-zero depth values


    return {
        "land_cover": ee.ImageCollection("ESA/WorldCover/v200").first().reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
        "solar": ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select("surface_solar_radiation_downwards_sum").mean().reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
        "temp": ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select("temperature_2m").mean().reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
        "precip": ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select("total_precipitation_sum").mean().reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
        "terrain": ee.Terrain.products(srtm).reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
        "flood_occur": flood_occurrence_sum.divide(n_years).reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
        "flood_duration_mean": flood_duration_sum.divide(n_years).reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
        "flood_depth_mean": nonzero_depth.mean().unmask(0).reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
        "flood_depth_max": flood_depth_collection.max().reproject(crs=COMMON_CRS, scale=COMMON_SCALE),
    }
