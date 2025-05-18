from shapely.strtree import STRtree
import numpy as np
import pandas as pd

from solmate_app.constants import GRID_TYPES_MAPPING


def create_STR_tree(geometry_vals):
    geom_list = list(geometry_vals)
    return STRtree(geom_list)

def extract_nearest_distance(geometry_vals):
    def nearest_distance(geom):
        """
        Find the nearest distance from a geometry to a list of geometries.
        """
        if geom is None or geom.is_empty:
            return np.nan
        str_tree = create_STR_tree(geometry_vals)
        nearest_idx = str_tree.nearest(geom)
        nearest_geom = str_tree.geometries.take(nearest_idx)

        return geom.distance(nearest_geom)
    return nearest_distance

def extract_dist_name(gdf, col_name):
    def nearest_distance_and_name(point):
        """
        Find the nearest distance from a geometry to a list of geometries and return the name of the nearest geometry.
        """
        if point is None or point.is_empty:
            return pd.Series([np.nan, np.nan])
        distances = gdf.geometry.distance(point)
        nearest_idx = distances.idxmin()
        nearest_name = gdf.loc[nearest_idx, col_name] if not pd.isnull(nearest_idx) else np.nan
        nearest_distance = distances.min() if not pd.isnull(nearest_idx) else np.nan
        return pd.Series([nearest_distance, nearest_name])
    
    return nearest_distance_and_name

def add_proximity_cols(df, gdf_points, static_layers):
    
    faults_geom = static_layers['faults_geom']
    gdf_residential = static_layers['gdf_residential']
    gdf_airports = static_layers['gdf_airports']
    gdf_main_roads = static_layers['gdf_main_roads']
    gdf_protected = static_layers['gdf_protected']
    gdf_kba = static_layers['gdf_kba']
    gdf_spug = static_layers['gdf_spug']
    gdf_grid = static_layers['gdf_grid']


    # Get proximity to faults
    df["fault_line_prox"] = gdf_points.geometry.apply(
    lambda point: faults_geom.distance(point).min()
    )

    # Get proximity to residential areas
    res_nearest_distance = extract_nearest_distance(gdf_residential.geometry.values)
    df["residential_prox"] = gdf_points.geometry.apply(res_nearest_distance)

    # Get proximity to airports
    air_nearest_distance = extract_nearest_distance(gdf_airports.geometry.values)
    df["airport_prox"] = gdf_points.geometry.apply(air_nearest_distance)

    # Get proximity to main roads
    roads_nearest_distance = extract_nearest_distance(gdf_main_roads.geometry.values)
    df["road_prox"] = gdf_points.geometry.apply(roads_nearest_distance)

    # Get proximity to protected areas and KBAs
    nearest_protected_area = extract_dist_name(gdf_protected, "NAME")
    df[["protected_area_prox", "nearest_protected_area"]] = gdf_points.geometry.apply(nearest_protected_area)

    nearest_kba = extract_dist_name(gdf_kba, "NatName")
    df[["kba_prox", "nearest_kba"]] = gdf_points.geometry.apply(nearest_kba)

    # Retain nearest protected area or KBA
    df["nearest_protected_area"] = df.apply(
        lambda row: row["nearest_protected_area"] if row["protected_area_prox"] < row["kba_prox"] else row["nearest_kba"],
        axis=1
    )

    # Retain minimum distance to protected area or KBA
    df["protected_area_prox"] = df.apply(
        lambda row: row["protected_area_prox"] if row["protected_area_prox"] < row["kba_prox"] else row["kba_prox"],
        axis=1
    )

    # Drop the KBA columns
    df = df.drop(columns=["kba_prox", "nearest_kba"])

    # Get proximity to grids
    nearest_grid = extract_dist_name(gdf_grid, "power")
    df[["grid_prox", "nearest_grid"]] = gdf_points.geometry.apply(nearest_grid)

    # Format the grid type
    df['nearest_grid'] = df['nearest_grid'].map(GRID_TYPES_MAPPING)

    # Get proximity to SPUG areas
    nearest_spug = extract_dist_name(gdf_spug, "adm4_en")
    df[["spug_prox", "nearest_spug"]] = gdf_points.geometry.apply(nearest_spug)

    # Check if in protected or SPUG area
    df["in_protected_area"] = df.apply(
        lambda row: "Yes" if row["protected_area_prox"] < 1e-6 else "No", axis=1
    )

    df["in_spug"] = df.apply(
        lambda row: "Yes" if row["spug_prox"] < 1e-6 else "No", axis=1
    )

    return df

