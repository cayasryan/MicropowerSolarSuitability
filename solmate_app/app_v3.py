import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import streamlit as st

from solmate_app.ui.home_page import show_home_page
from solmate_app.ui.results_page import show_results_page

st.set_page_config(
    page_title="SolMate – Solar Site Assessment Tool",
    page_icon="https://emojiterra.com/sun-with-face/",
    layout="wide",
)

# Initialize page variable
if "page" not in st.session_state:
    st.session_state.page = "home"
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "styled_df" not in st.session_state:
    st.session_state.styled_df = None




if st.session_state.page == "home":
    show_home_page()
elif st.session_state.page == "results":
    show_results_page()