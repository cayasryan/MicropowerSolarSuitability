import streamlit as st
import pandas as pd
from time import time
import os

from solmate_app.data_process.gee_process import init_gee
from solmate_app.data_process.loader import load_static_layers, load_gee_data
from solmate_app.data_process.utils import extract_coords
from solmate_app.assess.pipeline import assess
from solmate_app.ui.highlight import style_dataframe

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# For logo
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# -----------------------------------------------------------------------------
# HERO SECTION (HOME PAGE)
# -----------------------------------------------------------------------------
def show_home_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Use custom HTML for a styled, center-aligned hero section
    
        base64_logo_string = get_base64_image("solmate_logo.png")

        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="font-size: 5em;"><img src="data:image/png;base64,{logo}" width="150"> SolMate</h1>
                <h3 >Your smart companion for solar site assessment.</h3>
                <p style="font-size: 1.1em; color: #555;">
                    To get started, please provide the coordinates of the sites you want to assess.
                </p>
            </div>
            """.format(logo=base64_logo_string),
            unsafe_allow_html=True
        )


    # -----------------------------------------------------------------------------
    # FILE UPLOAD + START BUTTON (wrapped in a form)
    # -----------------------------------------------------------------------------
        with st.form("input_form", clear_on_submit=False):
            # File upload option
            st.markdown("📄 Option 1: Upload a CSV file containing 'Latitude' and 'Longitude':")
            uploaded_file = st.file_uploader(
                label="Input CSV file",
                label_visibility="collapsed",
                type=["csv"]
            )

            # Manual entry option
            st.markdown("📝 Option 2: Or enter coordinates manually:")
            default_data = pd.DataFrame({
                "Latitude": [None],
                "Longitude": [None]
            })
            manual_coords = st.data_editor(
                default_data,
                num_rows="dynamic",
                use_container_width=True
            )

            submitted = st.form_submit_button("Start Suitability Assessment")


    # If user hasn't clicked the button or hasn't uploaded a file, stop here
        if not submitted:
            st.stop()
        else:
            df = None

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"❌ Unable to read file: {e}")
                    st.stop()
                st.success("✔️ File uploaded successfully.")
            elif not manual_coords.dropna(how="any").empty:
                df = manual_coords.dropna(how="any")

                # Convert to float
                df['Latitude'] = df['Latitude'].astype(float)
                df['Longitude'] = df['Longitude'].astype(float)

                st.success("✔️ Using manually entered coordinates.")
            else:
                st.error("❌ Please upload a CSV file or enter at least one set of coordinates.")
                st.stop()

    # -----------------------------------------------------------------------------
    # PROCESS THE UPLOADED FILE
    # -----------------------------------------------------------------------------

        if df is not None:
            # Proceed with extract_coords, assess, etc.
            original_df, df, coords_exist = extract_coords(df)
            
            if not coords_exist:
                st.error("❌ Uploaded file must contain **Latitude** and **Longitude** columns.")
                st.stop()




    # -----------------------------------------------------------------------------
    # INITIALIZE DATA SOURCES
    # -----------------------------------------------------------------------------
    with col2:
        with st.spinner("Initializing Earth Engine and fetching data …"):
            warm_up_start = time()
            init_gee()
            static_layers = load_static_layers()
            gee_layers = load_gee_data()
            warm_up_end = time()
            os.write(1, f"Warm-up time: {warm_up_end - warm_up_start:.2f} seconds\n".encode()) 



    # -----------------------------------------------------------------------------
    # RUN THE ASSESSMENT
    # -----------------------------------------------------------------------------
    with col2:
        with st.spinner("Running suitability assessment …"):
            inf_start = time()
            df_pred = assess(df, static_layers, gee_layers)
            df_final, styled_df = style_dataframe(df_pred, original_df)
            inf_end = time()
            os.write(1, f"Assessment time: {120.0 + inf_end - inf_start:.2f} seconds\n".encode()) 

        st.success("Assessment complete! 🎉")

        st.session_state.df_final = df_final 
        st.session_state.styled_df = styled_df  
        st.session_state.page = "results"
        st.rerun() 