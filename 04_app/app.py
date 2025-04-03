import streamlit as st
import geemap.foliumap as geemap
import ee
import pandas as pd
import folium

# Initialize Earth Engine
try:
    ee.Initialize(project="micropower-app")
except Exception as e:
    st.error(f"Error initializing Earth Engine: {e}")

# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("Geospatial Feature Extraction using Google Earth Engine")

# Sidebar for File Upload and Layer Toggles
with st.sidebar:
    st.header("Upload Coordinates")
    uploaded_file = st.file_uploader("CSV File (Latitude, Longitude)", type=["csv"])

    st.header("Display Layers")
    show_land_cover = st.checkbox("Land Cover", value=False)
    show_solar = st.checkbox("Annual Solar Irradiance", value=False)
    show_temp = st.checkbox("Annual Surface Temperature", value=False)
    show_precip = st.checkbox("Annual Precipitation", value=False)
    if st.button("Show All Layers"):
        show_land_cover = show_solar = show_temp = show_precip = True

# Initialize Map
m = geemap.Map(center=[12.8797, 121.7740], zoom=6)

# Land Cover Data (ESA WorldCover 2021)
land_cover = ee.ImageCollection("ESA/WorldCover/v100").first()
land_cover_vis = {
    "bands": ["Map"],
    "min": 10,
    "max": 100,
    "palette": ["006400", "ffbb22", "ffff4c", "f096ff", "fa0000", "b4b4b4", "f0f0f0", "0064c8", "0096a0", "00cf75"],
}
if show_land_cover:
    m.addLayer(land_cover, land_cover_vis, "ESA WorldCover 2021")

# Climate Datasets
solar = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY").select("surface_solar_radiation_downwards").mean()
temp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY").select("temperature_2m").mean()
precip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY").select("total_precipitation").mean()

if show_solar:
    m.addLayer(solar, {"min": 50, "max": 300, "palette": ["yellow", "red"]}, "Solar Irradiance")
if show_temp:
    m.addLayer(temp, {"min": 270, "max": 310, "palette": ["blue", "red"]}, "Surface Temperature")
if show_precip:
    m.addLayer(precip, {"min": 0, "max": 300, "palette": ["white", "blue"]}, "Precipitation")

# Extract Feature Data Function
def extract_values(lat, lon):
    point = ee.Geometry.Point(lon, lat)
    extracted_values = []

    try:
        land_value = land_cover.sample(region=point, scale=30).first().get("Map").getInfo()
        solar_value = solar.sample(region=point, scale=1000).first().get("surface_solar_radiation_downwards").getInfo()
        temp_value = temp.sample(region=point, scale=1000).first().get("temperature_2m").getInfo()
        precip_value = precip.sample(region=point, scale=1000).first().get("total_precipitation").getInfo()
        
        land_labels = {10: "Tree Cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare/Sparse Veg", 70: "Snow/Ice", 80: "Water", 90: "Wetlands", 100: "Mangroves"}
        land_cover_label = land_labels.get(land_value, "Unknown")
    
        extracted_values = [land_cover_label, solar_value, temp_value, precip_value]

    except Exception as e:
        print(f"Error fetching values for ({lat}, {lon}): {e}")
        extracted_values = [None, None, None, None]

    return extracted_values



# Process Uploaded File
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "latitude" in df.columns and "longitude" in df.columns:
        st.success("✅ File Uploaded Successfully!")
        df[["Land Cover", "Solar Irradiance", "Surface Temp", "Precipitation"]] = df.apply(lambda row: extract_values(row["latitude"], row["longitude"]), axis=1, result_type="expand")
        
        # Display Table
        st.write("### Extracted Geospatial Data")
        st.dataframe(df)
        
        # Downloadable CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "geospatial_data.csv", "text/csv")
        
        # Add Markers to Map
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
                popup=f"Lat: {row['latitude']}, Lon: {row['longitude']}, Land Cover: {row['Land Cover']}\nSolar: {row['Solar Irradiance']} W/m²\nTemp: {row['Surface Temp']}K\nPrecip: {row['Precipitation']} mm"
            ).add_to(m)

# Show Map
m.to_streamlit(height=600)