import streamlit as st
import pandas as pd
import geemap.foliumap as geemap

from solmate_app.data_process.gee_process import init_gee
from solmate_app.data_process.loader import load_static_layers, load_gee_data
from solmate_app.data_process.utils import extract_coords
from solmate_app.assess.pipeline import assess
from solmate_app.ui.highlight import style_dataframe
from solmate_app.ui.map import add_markers



# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("SolMate")
st.subheader("Your smart companion for solar site assessment.")


init_gee()
static_layers = load_static_layers()
gee_layers = load_gee_data()


# Initialize Map
m = geemap.Map(center=[12.8797, 121.7740], zoom=6)


# Sidebar for File Upload and Legends
with st.sidebar:
    st.header("Upload Coordinates")
    uploaded_file = st.file_uploader("Upload a CSV file containing columns for Latitude and Longitude", type=["csv"])


# Process Uploaded File
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    original_df, df, coords_exist = extract_coords(df)

    if coords_exist:
        df_pred = assess(df, static_layers, gee_layers)
        df_final, styled_df = style_dataframe(df_pred, original_df)

        # Display Table
        st.dataframe(styled_df)
        
        # Downloadable CSV
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "solar_suitability.csv", "text/csv")

        # Add Markers to Map 
        add_markers(m, df_final)
                
    else:
        st.sidebar.error("❌ Please upload a CSV file with 'Latitude' and 'Longitude' columns.")

# Show Map
m.to_streamlit(height=600)