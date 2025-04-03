import streamlit as st
import leafmap.foliumap as leafmap
import ee

# Initialize Google Earth Engine
ee.Initialize(project="micropower-app")

# Define the Philippines boundary (GeoJSON)
philippines = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
    .filter(ee.Filter.eq("country_na", "Philippines"))

# Convert FeatureCollection to an image for visualization
philippines_image = philippines.style(**{
    "color": "0000FF",  # Blue boundary
    "width": 2,         # Line width
    "fillColor": "00000000"  # Transparent fill
})

# Create a Leafmap map centered on the Philippines
Map = leafmap.Map(center=[12.8797, 121.7740], zoom=6)

# Use addLayer() instead of add_ee_layer()
Map.addLayer(philippines_image, {}, "Philippines Boundary")

# Streamlit app layout
st.title("🗺️ Philippine Map with Google Earth Engine")
st.write("This is a simple Streamlit app displaying the Philippines boundary.")

# Display the Leafmap map in Streamlit
Map.to_streamlit(height=600)
