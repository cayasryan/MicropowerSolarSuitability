import folium

def add_markers(map, df):
    """
    Add markers to the map based on the suitability of the locations.
    
    Parameters:
    - map: The folium map object.
    - df: DataFrame containing latitude, longitude, suitability, and remarks.
    
    Returns:
    - None
    """
    for _, row in df.iterrows():
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
        ).add_to(map)