import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import json

import torch
import torch.nn as nn

import ee
import folium
import geemap.foliumap as geemap
import dask_geopandas as dgpd
from shapely.geometry import Point
import geopandas as gpd





# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("Site Suitability for Solar Micropowerplant Installation")

# Initialize Earth Engine
try:
    ee.Initialize(project="micropower-app")
except Exception as e:
    st.error(f"Error initializing Earth Engine: {e}")


# Initialize Map
m = geemap.Map(center=[12.8797, 121.7740], zoom=6)


# Sidebar for File Upload and Legends
with st.sidebar:
    st.header("Upload Coordinates")
    uploaded_file = st.file_uploader("CSV File (Latitude, Longitude)", type=["csv"])


st.sidebar.write("### Loading datasets...")
start_time = time.time()
gdf_protected = dgpd.read_parquet("data/protected_areas_reprojected.parquet").compute()
gdf_landcover = dgpd.read_parquet("data/land_cover_reprojected.parquet").compute()
st.sidebar.success(f"Data loaded in {time.time() - start_time:.2f} seconds")

## Load model
# st.write("### Loading model...")
start_time = time.time()
with open("model_params/autoencoder_params.json", "r") as f:
    loaded_params = json.load(f)

num_features = loaded_params["num_features"]
threshold_autoencoder = loaded_params["threshold_autoencoder"]

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Use Sigmoid for reconstruction between [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the model again
loaded_model = AutoEncoder(input_dim=num_features)

# Load the trained weights
loaded_model.load_state_dict(torch.load("model_weights/autoencoder_weights.pth"))

# Set the model to evaluation mode
loaded_model.eval()

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
land_cover_mapping = {
    1: "Terrestrial Forest",
    2: "Crop Areas",
    3: "Barren/Flatland",
    4: "Built-up",
    5: "Wetlands & Water Bodies"
}

# # Land Cover color mapping (highlighting Tree Cover & Crop Land)
# land_cover_colors = {
#     "Tree Cover": "background-color: #8B0000; color: white",  # Dark Red
#     "Cropland": "background-color: #FF4500; color: white",  # Orange Red
# }


def assess_suitability(df):
    # Convert DataFrame to GeoDataFrame
    print("Converting DataFrame to GeoDataFrame...")
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf_points = gdf_points.to_crs(epsg=32651)

    # Check if points are inside protected areas
    print("Checking if points are inside protected areas...")
    gdf_points["in_protected_area"] = gdf_points.sjoin(gdf_protected, how="left", predicate="intersects")['index_right'].notnull()

    # Get land cover type (assuming land cover GeoDataFrame has a 'land_type' column)
    print("Getting land cover type...")
    gdf_points = gdf_points.sjoin(gdf_landcover[['geometry', 'class_id']], how="left", predicate="intersects")

    st.sidebar.success("✅ Features retrieved successfully!")
    st.sidebar.write("### Assessing suitability...")

    # Drop unnecessary index_right column from spatial join
    gdf_points = gdf_points.drop(columns=['index_right'])

    # Define all possible land cover classes
    all_classes = [1, 2, 3, 4, 5]

    # Ensure all classes appear in one-hot encoding
    gdf_points['class_id'] = pd.Categorical(gdf_points['class_id'], categories=all_classes)
    df_encoded = pd.get_dummies(gdf_points[['class_id']], columns=['class_id'], prefix='landcover_class').astype(int)


    # Convert boolean 'in_predicted_area' to 1/0
    df_encoded['in_preotected_area'] = gdf_points['in_protected_area'].astype(int)

    df_encoded_array = df_encoded.to_numpy()

    df_encoded_tensor = torch.tensor(df_encoded_array, dtype=torch.float32)

    with torch.no_grad():
        test_reconstruction = loaded_model(df_encoded_tensor).numpy()
        test_reconstruction_error = np.mean(np.square(df_encoded_array - test_reconstruction), axis=1)
        test_anomalies_autoencoder = np.where(test_reconstruction_error > threshold_autoencoder, 1, 0)
    
    # Map class_id to land cover names
    df['land_cover'] = gdf_points['class_id'].map(land_cover_mapping)

    # Convert 'in_predicted_area' to 1/0
    df['in_preotected_area'] = gdf_points['in_protected_area']

    df['suitability'] = np.where(test_anomalies_autoencoder == 1, "Likely Unsuitable", "Suitable")

    return df



# Process Uploaded File
if uploaded_file is not None:
    start_time = time.time()
    df = pd.read_csv(uploaded_file)
    if "latitude" in df.columns and "longitude" in df.columns:
        st.sidebar.success("✅ File Uploaded Successfully!")

        st.sidebar.write("### Querying Location Features...")

        df_pred = assess_suitability(df)
        st.sidebar.success("✅ Suitability assessment completed!")

        # Function to apply styling
        def highlight_suitability(val):
            return suitability_colors.get(val, "")
        
        # styled_df = df_pred.style.applymap(highlight_suitability, subset=["Suitability"])
        # Display Table
        st.write("### Extracted Geospatial Data")
        st.dataframe(df_pred)
        
        # Downloadable CSV
        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "solar_suitability.csv", "text/csv")


        # Add Markers to Map
        for _, row in df.iterrows():
            suitability = row['suitability']

            if suitability == "Suitable":
                color = "green"
            else:
                color = "red"


            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Lat: {row['latitude']}, Lon: {row['longitude']}, {row['suitability']}"
            ).add_to(m)

# Show Map
m.to_streamlit(height=600)


        

