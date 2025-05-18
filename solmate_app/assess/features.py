import geopandas as gpd
from shapely.geometry import Point

from solmate_app.data_process.gee_process import extract_gee_values
from solmate_app.data_process.proximity import add_proximity_cols


def extract_features(df, static_layers, gee_layers):
    """
    Extracts features from the given DataFrame using the provided layers.

    Args:
        df (pd.DataFrame): The input DataFrame containing coordinates.
        static_layers (dict): Dictionary containing static layers.
        gee_layers (dict): Dictionary containing GEE layers.
    Returns:
        pd.DataFrame: The DataFrame with extracted features.
    """

    # Convert DataFrame to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Extract GEE values
    df = extract_gee_values(df, gdf_points, gee_layers)

    gdf_points = gdf_points.to_crs(epsg=32651)

    # Add proximity columns
    df = add_proximity_cols(df, gdf_points, static_layers)

    return df