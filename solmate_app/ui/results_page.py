import streamlit as st
import geemap.foliumap as geemap

from solmate_app.ui.map import add_markers
from solmate_app.constants import HIGHLIGHT_COLORS


# -----------------------------------------------------------------------------
# RESULTS — MAP
# -----------------------------------------------------------------------------
def show_results_page():
    st.title("Assessment Results")

    # Display results (map, tables, etc)
    df_final = st.session_state.df_final
    styled_df = st.session_state.styled_df

    # Create 2/3 and 1/3 layout
    left_col, right_col = st.columns([2, 1])

    # Set default centre near Philippines if at least one site can't be used
    with left_col:
        centre = [12.8797, 121.7740]
        if not df_final.empty:
            first_row = df_final.iloc[0]
            if "Latitude" in first_row and "Longitude" in first_row:
                centre = [first_row["Latitude"], first_row["Longitude"]]

        m = geemap.Map(center=centre, zoom=6)
        add_markers(m, df_final)
        m.to_streamlit(height=600)

    # SUMMARY in right column
    with right_col:
        suitable_count = df_final[df_final["Suitability"] == "Suitable"].shape[0]
        unsuitable_count = df_final[df_final["Suitability"] != "Suitable"].shape[0]


        def render_summary(suitable_count, unsuitable_count):
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 1rem;
                    max-width: 100%;
                    font-family: Arial, sans-serif;
                ">
                    <strong style="font-size: 2rem;">Site Suitability Breakdown</strong><br><br>
                    <span style="color: {HIGHLIGHT_COLORS['summary_suitable']}; font-size: 2rem; font-weight: 600;">{suitable_count} Suitable Sites</span><br>
                    <span style="color: {HIGHLIGHT_COLORS['summary_unsuitable']}; font-size: 2rem; font-weight: 600;">{unsuitable_count} Unsuitable Sites</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_summary(suitable_count, unsuitable_count)

        

    # -----------------------------------------------------------------------------
    # RESULTS — TABLE + DOWNLOAD
    # -----------------------------------------------------------------------------
    st.markdown("### 📊 Full Site Information")
    st.dataframe(styled_df, use_container_width=True)


    csv = df_final.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download results as CSV", csv, "solar_suitability.csv")

    if st.button("🔁 Start Another Analysis"):
        st.session_state.page = "home"
        st.session_state.df_final = None
        st.session_state.styled_df = None
        st.rerun() 