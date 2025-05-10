import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import json

import os
from glob import glob

import torch
import torch.nn as nn

import ee
import folium
import geemap.foliumap as geemap
import geopandas as gpd
import dask_geopandas as dgpd
from shapely.geometry import Point
import geopandas as gpd

from shapely.strtree import STRtree
import numpy as np



# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("SolMate")
st.subheader("Your smart companion for solar site assessment.")

service_account_info = {
    "type": st.secrets["earthengine"]["type"],
    "project_id": st.secrets["earthengine"]["project_id"],
    "private_key_id": st.secrets["earthengine"]["private_key_id"],
    "private_key": st.secrets["earthengine"]["private_key"],
    "client_email": st.secrets["earthengine"]["client_email"],
    "client_id": st.secrets["earthengine"]["client_id"],
    "auth_uri": st.secrets["earthengine"]["auth_uri"],
    "token_uri": st.secrets["earthengine"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["earthengine"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["earthengine"]["client_x509_cert_url"],
    "universe_domain": st.secrets["earthengine"]["universe_domain"]
}

credentials = ee.ServiceAccountCredentials(service_account_info["client_email"], key_data=service_account_info)

# Initialize Earth Engine
try:
    ee.Initialize(credentials)
except Exception as e:
    st.error(f"Error initializing Earth Engine: {e}")


# Initialize Map
m = geemap.Map(center=[12.8797, 121.7740], zoom=6)


# Sidebar for File Upload and Legends
with st.sidebar:
    st.header("Upload Coordinates")
    uploaded_file = st.file_uploader("Upload a CSV file containing columns for Latitude and Longitude", type=["csv"])


### LOAD DATASETS ------------------------------------------------------------------------------------------------------------------

st.sidebar.write("### Loading datasets...")
start_time = time.time()
gdf_protected = dgpd.read_parquet("01_processed_data/protected_areas_reprojected.parquet").compute()
gdf_kba = gpd.read_file("01_processed_data/philippines_kba.geojson")
# gdf_landcover = dgpd.read_parquet("../01_processed_data/land_cover_reprojected.parquet").compute()
# gdf_flood_5 = dgpd.read_parquet("../01_processed_data/flood_risk/FloodRisk_5yr_reprojected.parquet").compute()
# gdf_flood_25 = dgpd.read_parquet("../01_processed_data/flood_risk/FloodRisk_25yr_reprojected.parquet").compute()
# gdf_flood_100 = dgpd.read_parquet("../01_processed_data/flood_risk/FloodRisk_100yr_reprojected.parquet").compute()

faults_geom = gpd.read_file("01_processed_data/faults_ph_geometry.geojson")

residential_1 = gpd.read_file("01_processed_data/residential_areas_part1.geojson")
residential_2 = gpd.read_file("01_processed_data/residential_areas_part2.geojson")
residential = pd.concat([residential_1, residential_2], ignore_index=True)


land_cover = ee.ImageCollection("ESA/WorldCover/v200").first()

# Climate Datasets
solar = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select("surface_solar_radiation_downwards_sum").mean()
temp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select("temperature_2m").mean()
precip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select("total_precipitation_sum").mean()

# Flood Data
flood_collection = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1") \
    .select("flooded") \
    .map(lambda img: img.unmask(0))  # Replace null/masked values with 0

flood = flood_collection.sum()# flood_depth = ee.ImageCollection("JRC/CEMS_GLOFAS/FloodHazard/v1").select("depth").mean()

flood_depth_collection = ee.ImageCollection("JRC/CEMS_GLOFAS/FloodHazard/v1") \
    .select("depth") \
    .map(lambda img: img.unmask(0))  # Replace null/masked values with 0
flood_depth_mean = flood_depth_collection.mean()
flood_depth_max = flood_depth_collection.max()

# flood_depth = ee.ImageCollection("JRC/CEMS_GLOFAS/FloodHazard/v1").select("depth").max()


# flood_max = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").select("flooded").max()
# flood_mean = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").select("flooded").mean()
# flood_dur_max = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").select("duration").max()
# flood_dur_mean = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").select("duration").mean()

#-----------------------------------------------------------------------------------------------------------------------------------


### LOAD AND PROCESS FLOOD RISK FILES -------------------------------------------------------------

def merge_parquet_folder(folder_path):
    # Get all .parquet files in the folder
    parquet_files = sorted(glob(os.path.join(folder_path, "*.parquet")))

    # Use list comprehension to read each file into a GeoDataFrame
    gdfs = [gpd.read_parquet(f) for f in parquet_files]

    # Concatenate into one GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return merged_gdf

# gdf_flood_5 = merge_parquet_folder("01_processed_data/flood_5_split_parquet")
# gdf_flood_25 = merge_parquet_folder("01_processed_data/flood_25_split_parquet")
# gdf_flood_100 = merge_parquet_folder("01_processed_data/flood_100_split_parquet")

st.sidebar.success(f"Data loaded in {time.time() - start_time:.2f} seconds")
# -----------------------------------------------------



# PROCESS RESIDENTIAL AREAS ------------------------------------------------------------------------------------------------------------------------------------

# Create a spatial index from residential geometries
res_geom_list = list(residential.geometry.values)  # ensure it's a plain list of geometries
res_tree = STRtree(res_geom_list)

# For each site, find the nearest residential polygon and compute distance
def nearest_distance(site_geom):
    if site_geom is None or site_geom.is_empty:
        return np.nan
    nearest_idx = res_tree.nearest(site_geom)
    nearest_geom = res_tree.geometries.take(nearest_idx)

    return site_geom.distance(nearest_geom)

# -------------------------------------------------------


# PROCESS GEE DATA ------------------------------------------------------------------------------------------------------------------------------------

def create_feature_collection(df):
    features = [
        ee.Feature(ee.Geometry.Point(row['longitude'], row['latitude']), {'id': idx})
        for idx, row in df.iterrows()
    ]
    return ee.FeatureCollection(features)


# Extract Feature Data Function
def extract_GEE_values(df):
    fc_points = create_feature_collection(df)

    st.sidebar.write("Getting land cover info...")
    st.sidebar.write("Extracting info on climate and risk factors...")

    sampled = land_cover \
    .addBands(solar.rename("solar")) \
    .addBands(temp.rename("temp")) \
    .addBands(precip.rename("precip")) \
    .addBands(flood.rename("flood")) \
    .addBands(flood_depth_mean.rename("flood_mean")) \
    .addBands(flood_depth_max.rename("flood_max")) \
    .sampleRegions(collection=fc_points, scale=10, geometries=True)


    # .addBands(flood_depth_mean.rename("flood_mean")) \
    # .addBands(flood_depth_max.rename("flood_max")) \

     # .addBands(flood_max.rename("flood_max")) \
    # .addBands(flood_mean.rename("flood_mean")) \
    # .addBands(flood_dur_max.rename("flood_dur_max")) \
    # .addBands(flood_dur_mean.rename("flood_dur_mean")) \

    results = sampled.getInfo()

    extracted = []
    for f in results['features']:
        props = f['properties']
        extracted.append({
            'id': props['id'],
            'land_cover': props.get('Map'),  # land cover code
            'Monthly Surface Solar Radiation (J/m²)': props.get('solar'), # Monthly Surface Solar Radiation (J/m²)
            'Mean 2m Temperature (°C)': props.get('temp') - 273.15, # Mean 2m Temperature (K)
            'Mean Monthly Precipitation (m)': props.get('precip'), # Mean Monthly Precipitation (m)
            'Flood Extent History': props.get('flood'), 
            'Mean Flood Depth (m)': props.get('flood_mean'), # Mean Flood Depth (m)
            'Max Flood Depth (m)': props.get('flood_max'), # Max Flood Depth (m)
            # 'Mean Flood Duration (days)': props.get('flood_dur_mean'), # Mean Flood Duration (days)
            # 'Max Flood Duration (days)': props.get('flood_dur_max'), # Max Flood Duration (days)
            # 'Mean Flood Extent (%)': props.get('flood_mean'), 
            # 'Max Flood Extent (%)': props.get('flood_max')
        })

    df['id'] = df.index
    extracted = pd.DataFrame(extracted)

    # Merge the extracted data with the original DataFrame
    df_results = pd.merge(df, extracted, on='id', how='left')
    # Drop the 'id' column
    df_results = df_results.drop(columns=['id'])

    # Decoding land cover codes
    land_labels = {
        10: "Tree Cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
        50: "Built-up", 60: "Bare/Sparse Veg", 70: "Snow/Ice", 80: "Water",
        90: "Wetlands", 95: "Mangroves", 100: "Moss & Lichen"
    }

    df_results['Land Cover'] = df_results['land_cover'].map(land_labels)
    # drop land cover code
    df_results = df_results.drop(columns=['land_cover'])

    return df_results


# -------------------------------------------------------------------------------------------------------------------------------------


## Load model
# st.write("### Loading model...")
# start_time = time.time()
# with open("../04_app/model_params/autoencoder_params.json", "r") as f: # CHANGE DIR
#     loaded_params = json.load(f)

# num_features = loaded_params["num_features"]
# threshold_autoencoder = loaded_params["threshold_autoencoder"]

# class AutoEncoder(nn.Module):
#     def __init__(self, input_dim):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, input_dim),
#             nn.Sigmoid()  # Use Sigmoid for reconstruction between [0, 1]
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # Instantiate the model again
# loaded_model = AutoEncoder(input_dim=num_features)

# # Load the trained weights
# loaded_model.load_state_dict(torch.load("../04_app/model_weights/autoencoder_weights.pth"))

# # Set the model to evaluation mode
# loaded_model.eval()

# st.write(f"Model loaded in {time.time() - start_time:.2f} seconds")



# Define suitability categories
suitability_categories = [
    "Likely Unsuitable",
    # "Highly Unsuitable",
    # "Moderately Unsuitable",
    # "Marginally Suitable",
    # "Moderately Suitable",
    "Suitable"
]

suitability_colors = {
    "Likely Unsuitable": "background-color: #8B0000; color: white",  # Dark Red
    # "Likely Unsuitable": "background-color: #FF4500; color: white",  # Orange Red
    # "Moderately Unsuitable": "background-color: #FFA500; color: black",  # Orange
    # "Marginally Suitable": "background-color: #FFFF00; color: black",  # Yellow
    # "Moderately Suitable": "background-color: #ADFF2F; color: black",  # Green Yellow
    "Suitable": "background-color: #008000; color: white"  # Green
}

# Mapping dictionary
# land_cover_mapping = {
#     1: "Terrestrial Forest",
#     2: "Crop Areas",
#     3: "Barren/Flatland",
#     4: "Built-up",
#     5: "Wetlands & Water Bodies"
# }

# flood_risk_mapping = {
#     0: "No Risk",
#     1: "Low",
#     2: "Medium",
#     3: "High"
# }

# # Land Cover color mapping (highlighting Tree Cover & Crop Land)
# land_cover_colors = {
#     "Tree Cover": "background-color: #8B0000; color: white",  # Dark Red
#     "Cropland": "background-color: #FF4500; color: white",  # Orange Red
# }


def assess_suitability(df):
    start_time = time.time()

    # Convert DataFrame to GeoDataFrame
    print("Converting DataFrame to GeoDataFrame...")
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Extract GEE values
    df = extract_GEE_values(gdf_points)

    gdf_points = gdf_points.to_crs(epsg=32651)


    # Get Min Distance to Fault Lines
    st.sidebar.write("Calculating distance to nearest fault line...")
    df["Min. Distance to Fault Line (m)"] = gdf_points.geometry.apply(
    lambda point: faults_geom.distance(point).min()
    )

    st.sidebar.write("Calculating distance to nearest residential area...")
    df['Min. Distance to Residential Areas (m)'] = gdf_points.geometry.apply(nearest_distance)


    # drop geometry
    df = df.drop(columns=['geometry'])

    # Check if points are inside protected areas
    st.sidebar.write("Checking for restricted areas...")
    gdf_points = gdf_points.reset_index(drop=True)
    joined_gdf = gdf_points.sjoin(gdf_protected, how="left", predicate="intersects")
    joined_gdf = joined_gdf.sort_values(by='index_right', ascending=False)
    joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep='first')]
    gdf_points["in_protected_area"] = joined_gdf['index_right'].notnull()


    gdf_points = gdf_points.reset_index(drop=True)
    joined_gdf = gdf_points.sjoin(gdf_kba, how="left", predicate="intersects")
    joined_gdf = joined_gdf.sort_values(by='index_right', ascending=False)
    joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep='first')]
    gdf_points["in_KBA"] = joined_gdf['index_right'].notnull()


    # Get land cover type (assuming land cover GeoDataFrame has a 'land_type' column)
    # print("Getting land cover type...")
    # gdf_points = gdf_points.sjoin(gdf_landcover[['geometry', 'class_id']], how="left", predicate="intersects")

    # Drop unnecessary index_right column from spatial join
    # gdf_points = gdf_points.drop(columns=['index_right'])

    # print("Getting flood risk...")
    # gdf_points = gdf_points.sjoin(gdf_flood_5[['geometry', 'FloodRisk']], how="left", predicate="intersects").fillna({'FloodRisk': 0}).rename(columns={'FloodRisk': 'FloodRisk_5'}).drop(columns=['index_right'], errors='ignore')
    # gdf_points = gdf_points.sjoin(gdf_flood_25[['geometry', 'FloodRisk']], how="left", predicate="intersects").fillna({'FloodRisk': 0}).rename(columns={'FloodRisk': 'FloodRisk_25'}).drop(columns=['index_right'], errors='ignore')
    # gdf_points = gdf_points.sjoin(gdf_flood_100[['geometry', 'FloodRisk']], how="left", predicate="intersects").fillna({'FloodRisk': 0}).rename(columns={'FloodRisk': 'FloodRisk_100'}).drop(columns=['index_right'], errors='ignore')

    st.sidebar.success(f"Features retrieved in {time.time() - start_time:.2f} seconds")



    st.sidebar.write("### Assessing suitability...")
    

    # Define all possible land cover classes
    # all_classes = [1, 2, 3, 4, 5]

    # # Ensure all classes appear in one-hot encoding
    # gdf_points['class_id'] = pd.Categorical(gdf_points['class_id'], categories=all_classes)
    # df_encoded = pd.get_dummies(gdf_points[['class_id']], columns=['class_id'], prefix='landcover_class').astype(int)


    # Convert boolean 'in_predicted_area' to 1/0
    # df_encoded['in_preotected_area'] = gdf_points['in_protected_area'].astype(int)

    # df_encoded_array = df_encoded.to_numpy()

    # df_encoded_tensor = torch.tensor(df_encoded_array, dtype=torch.float32)

    # with torch.no_grad():
    #     test_reconstruction = loaded_model(df_encoded_tensor).numpy()
    #     test_reconstruction_error = np.mean(np.square(df_encoded_array - test_reconstruction), axis=1)
    #     test_anomalies_autoencoder = np.where(test_reconstruction_error > threshold_autoencoder, 1, 0)
    

    # Convert 'in_predicted_area' to 1/0
    df['in_protected_area'] = gdf_points['in_protected_area']
    df['in_KBA'] = gdf_points['in_KBA']

    # Get Flood Risk
    # df['FloodRisk_5yr'] = gdf_points['FloodRisk_5'].astype(int).map(flood_risk_mapping)
    # df['FloodRisk_25yr'] = gdf_points['FloodRisk_25'].astype(int).map(flood_risk_mapping)
    # df['FloodRisk_100yr'] = gdf_points['FloodRisk_100'].astype(int).map(flood_risk_mapping)

    # Map class_id to land cover names
    # df['land_cover'] = gdf_points['class_id'].map(land_cover_mapping)

    # df['suitability'] = np.where(test_anomalies_autoencoder == 1, "Likely Unsuitable", "Suitable")

    # Randomly assign suitability for demonstration
    # df['suitability'] = np.random.choice(suitability_categories, size=len(df))

    def classify_suitability_recoms(row):
        unsuitable_land_covers = [
            "Snow/Ice", "Water", "Wetlands", "Mangroves", "Cropland"
        ]
        
        issues = []

        if row['in_protected_area'] or row['in_KBA']:
            issues.append("In Restricted Area")
        
        if row['Flood Extent History'] > 0.4 or row['Mean Flood Depth (m)'] > 0.5 or row['Max Flood Depth (m)'] > 1:
            issues.append("High Flood Risk")

        if row['Min. Distance to Fault Line (m)'] < 5:
            issues.append("Near Fault Line")

        if row['Min. Distance to Residential Areas (m)'] < 10:
            issues.append("Near Residential Areas")

        if row['Monthly Surface Solar Radiation (J/m²)'] < 18000:
            issues.append("Low Solar Radiation")

        if row['Mean 2m Temperature (°C)'] < 0:
            issues.append("Low Surface Temperature")

        if row['Mean 2m Temperature (°C)'] > 45:
            issues.append("High Surface Temperature")

        if row['Mean Monthly Precipitation (m)'] > 0.5:
            issues.append("High Precipitation")

        if row['Land Cover'] in unsuitable_land_covers:
            issues.append(f"Land Cover is {row['Land Cover']}")

        if issues:
            return ['Likely Unsuitable', "; ".join(issues)]
        else:
            return ['Suitable', "No major issues detected."]

    
    df[['Suitability', 'Recommendation']] = df.apply(
        lambda row: pd.Series(classify_suitability_recoms(row)),
        axis=1
    )

    


    rename_mapping = {
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'in_protected_area': 'In Protected Area?',
        'in_KBA': 'In KBA?',
        # 'FloodRisk_5yr': 'Flood Risk (5-year)',
        # 'FloodRisk_25yr': 'Flood Risk (25-year)',
        # 'FloodRisk_100yr': 'Flood Risk (100-year)',
        # 'land_cover': 'Land Cover',
        # 'suitability': 'Suitability',
    }

    df = df.rename(columns=rename_mapping)

    
    return df



# Process Uploaded File
if uploaded_file is not None:
    start_time = time.time()
    df = pd.read_csv(uploaded_file)

    # Create a lowercase-to-original mapping
    col_map = {col.lower(): col for col in df.columns}

    # Rename if needed
    if 'latitude' in col_map and 'longitude' in col_map:
        df = df.rename(columns={
            col_map['latitude']: 'latitude',
            col_map['longitude']: 'longitude'
        })

    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.sidebar.success("✅ File Uploaded Successfully!")

        st.sidebar.write("### Querying Location Features...")

        df_pred = assess_suitability(df)
        st.sidebar.success("✅ Suitability assessment completed!")

        

        # Function to apply styling
        # def highlight_suitability_row(row):
        #     suitability_colors = {
        #         "Suitable": "background-color: #d4edda",           # Light green
        #         "Likely Unsuitable": "background-color: #f8d7da",  # Light red
        #         # "Unsuitable": "background-color: #f5c6cb",         # Deeper red
        #     }

        #     color = suitability_colors.get(row['Suitability'], '')
        #     return {
        #         'Suitability': color,
        #         'Recommendation': color  # Apply the same color regardless of status
        #     }
        
        # # Define all colors in one place
        # HIGHLIGHT_COLORS = {
        #     "unsuitable_feature": "background-color: #f8d7da",  # Light red
        #     "suitable_feature": "background-color: #d4edda",    # Light green (optional)
        # }

        # def highlight_unsuitable_features(row):
        #     highlight = [''] * len(row)
        #     columns = list(row.index)

        #     # Only process if not Suitable
        #     if row['Suitability'] == "Suitable":
        #         return highlight

        #     # Feature-specific conditions
        #     if "Protected Area" in row['Recommendation']:
        #         if row.get('in_protected_area', False):
        #             highlight[columns.index('in_protected_area')] = HIGHLIGHT_COLORS["unsuitable_feature"]
        #         if row.get('in_KBA', False):
        #             highlight[columns.index('in_KBA')] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     if "Flood Risk" in row['Recommendation']:
        #         flood_thresholds = {
        #             'Flood Extent History': 0.4,
        #             'Mean Flood Depth (m)': 0.5,
        #             'Max Flood Depth (m)': 1.0,
        #         }
        #         for col, threshold in flood_thresholds.items():
        #             if row.get(col, 0) > threshold:
        #                 highlight[columns.index(col)] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     if "Near Fault Line" in row['Recommendation']:
        #         if row.get('Min. Distance to Fault Line (m)', float('inf')) < 5:
        #             highlight[columns.index('Min. Distance to Fault Line (m)')] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     if "Near Residential Areas" in row['Recommendation']:
        #         if row.get('Min. Distance to Residential Areas (m)', float('inf')) < 5:
        #             highlight[columns.index('Min. Distance to Residential Areas (m)')] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     if "Low Solar Radiation" in row['Recommendation']:
        #         if row.get('Monthly Surface Solar Radiation (J/m²)', float('inf')) < 18000:
        #             highlight[columns.index('Monthly Surface Solar Radiation (J/m²)')] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     if "Low Surface Temperature" in row['Recommendation']:
        #         if row.get('Mean 2m Temperature (K)', 0) < 298:
        #             highlight[columns.index('Mean 2m Temperature (K)')] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     if "High Surface Temperature" in row['Recommendation']:
        #         if row.get('Mean 2m Temperature (K)', 0) > 350:
        #             highlight[columns.index('Mean 2m Temperature (K)')] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     if "High Precipitation" in row['Recommendation']:
        #         if row.get('Mean Monthly Precipitation (m)', 0) > 0.5:
        #             highlight[columns.index('Mean Monthly Precipitation (m)')] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     if "Land cover is" in row['Recommendation']:
        #         unsuitable_land_covers = [
        #             "Built-up", "Snow/Ice", "Water", "Wetlands", "Mangroves", "Tree Cover", "Cropland"
        #         ]
        #         if row.get('Land Cover') in unsuitable_land_covers:
        #             highlight[columns.index('Land Cover')] = HIGHLIGHT_COLORS["unsuitable_feature"]

        #     return highlight




        def highlight_suitability_and_features(row):
            # Define a dictionary of highlight colors
            highlight_colors = {
                'suitable': 'background-color: #d4edda',             # Green-ish
                'likely_unsuitable': 'background-color: #f8d7da',    # Light red
                'unsuitable': 'background-color: #f5c6cb',           # Darker red
                'recommendation_suitable': 'background-color: #d4edda',  # Green-ish for recommendations when suitable
                'recommendation_unsuitable': 'background-color: #f8d7da',  # Light red for unsuitable recommendations
                'highlight_affected_feature': 'background-color: #f8d7da'  # Light red for features causing unsuitability
            }

            # Initialize an empty list for cell styles
            highlight = [''] * len(row)
            
            # Define columns to style
            columns = list(row.index)

            # Highlight suitability column
            if row['Suitability'] == 'Suitable':
                highlight[columns.index('Suitability')] = highlight_colors['suitable']
            elif row['Suitability'] == 'Likely Unsuitable':
                highlight[columns.index('Suitability')] = highlight_colors['likely_unsuitable']
            elif row['Suitability'] == 'Unsuitable':
                highlight[columns.index('Suitability')] = highlight_colors['unsuitable']

            # Highlight recommendation column based on suitability
            if row['Suitability'] != 'Suitable':
                highlight[columns.index('Remarks')] = highlight_colors['recommendation_unsuitable']
            else:
                highlight[columns.index('Remarks')] = highlight_colors['recommendation_suitable']

            # Feature-specific highlighting for unsuitable factors
            if "Restricted Area" in row['Remarks']:
                if row.get('In Protected Area?', False):
                    highlight[columns.index('In Protected Area?')] = highlight_colors['highlight_affected_feature']
                if row.get('In KBA?', False):
                    highlight[columns.index('In KBA?')] = highlight_colors['highlight_affected_feature']

            if "Flood Risk" in row['Remarks']:
                flood_thresholds = {
                    'Flood Extent History': 0.4,
                    'Mean Flood Depth (m)': 0.5,
                    'Max Flood Depth (m)': 1.0,
                }
                for col, threshold in flood_thresholds.items():
                    if row.get(col, 0) > threshold:
                        highlight[columns.index(col)] = highlight_colors['highlight_affected_feature']

            if "Near Fault Line" in row['Remarks']:
                if row.get('Min. Distance to Fault Line (m)', float('inf')) < 5:
                    highlight[columns.index('Min. Distance to Fault Line (m)')] = highlight_colors['highlight_affected_feature']

            if "Near Residential Areas" in row['Remarks']:
                if row.get('Min. Distance to Residential Areas (m)', float('inf')) < 10:
                    highlight[columns.index('Min. Distance to Residential Areas (m)')] = highlight_colors['highlight_affected_feature']

            if "Low Solar Radiation" in row['Remarks']:
                if row.get('Monthly Surface Solar Radiation (J/m²)', float('inf')) < 18000:
                    highlight[columns.index('Monthly Surface Solar Radiation (J/m²)')] = highlight_colors['highlight_affected_feature']

            if "Low Surface Temperature" in row['Remarks']:
                if row.get('Mean 2m Temperature (°C)', 0) < 0:
                    highlight[columns.index('Mean 2m Temperature (K)')] = highlight_colors['highlight_affected_feature']

            if "High Surface Temperature" in row['Remarks']:
                if row.get('Mean 2m Temperature (°C)', 0) > 45:
                    highlight[columns.index('Mean 2m Temperature (K)')] = highlight_colors['highlight_affected_feature']

            if "High Precipitation" in row['Remarks']:
                if row.get('Mean Monthly Precipitation (m)', 0) > 0.5:
                    highlight[columns.index('Mean Monthly Precipitation (m)')] = highlight_colors['highlight_affected_feature']

            if "Land Cover is" in row['Remarks']:
                unsuitable_land_covers = [
                    "Snow/Ice", "Water", "Wetlands", "Mangroves", "Cropland"
                ]
                if row.get('Land Cover') in unsuitable_land_covers:
                    highlight[columns.index('Land Cover')] = highlight_colors['highlight_affected_feature']

            return highlight



        # styled_df = df_pred.style.apply(
        #     highlight_suitability_row, axis=1, subset=["Suitability", "Recommendation"]
        # )

        # styled_df = styled_df.style.apply(highlight_unsuitable_features, axis=1)

        cols_to_front = ['Latitude','Longitude','Suitability', 'Recommendation', "In Protected Area?", "In KBA?", 'Land Cover']
        df_pred = df_pred[cols_to_front + [col for col in df_pred.columns if col not in cols_to_front]]

        df_pred = df_pred.rename(columns={'Recommendation': 'Remarks'})

        styled_df = df_pred.style.apply(highlight_suitability_and_features, axis=1)

        




        # Display Table
        st.write("### Extracted Geospatial Data")
        st.dataframe(styled_df)
        
        # Downloadable CSV
        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "solar_suitability.csv", "text/csv")


        # Add Markers to Map
        for _, row in df_pred.iterrows():
            suitability = row['Suitability']

            if suitability == "Suitable":
                color = "green"
            else:
                color = "red"

            popup_html = (
                f"<b>Latitude:</b> {row['Latitude']}<br>"
                f"<b>Longitude:</b> {row['Longitude']}<br>"
                f"<b>Suitability:</b> {row['Suitability']}<br>"
                f"<b>Remarks:</b> {row['Remarks']}"
            )

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(m)
    
    else:
        st.sidebar.error("❌ Please upload a CSV file with 'Latitude' and 'Longitude' columns.")

# Show Map
m.to_streamlit(height=600)


        

