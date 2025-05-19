import json
import ee
import pandas as pd
import streamlit as st


from solmate_app.constants import LAND_LABELS


def init_gee():
    # Initialize Earth Engine
    try:
        service_account_info = dict(st.secrets["earthengine"])
        json_key = json.dumps(service_account_info)
        credentials = ee.ServiceAccountCredentials(
            email=service_account_info["client_email"],
            key_data=json_key
        )

        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Earth Engine: {e}")
        return False

def create_feature_collection(df):
    features = [
        ee.Feature(ee.Geometry.Point(row['longitude'], row['latitude']), {'id': idx})
        for idx, row in df.iterrows()
    ]
    return ee.FeatureCollection(features)

# Function to extract values per point with reduceRegion (allows nulls)
def make_extractor(gee_data):
    def extract_bands_to_feature(point):
        reducers = ee.Reducer.first()  # or ee.Reducer.mean() if averaging

        land_cover = gee_data['land_cover']
        solar = gee_data['solar']
        temp = gee_data['temp']
        precip = gee_data['precip']
        flood_occur = gee_data['flood_occur']
        flood_duration_mean = gee_data['flood_duration_mean']
        flood_depth_mean = gee_data['flood_depth_mean']
        flood_depth_max = gee_data['flood_depth_max']
        terrain = gee_data['terrain']

        # Combine all the bands into a single image
        combined = land_cover \
            .addBands(solar.rename("solar")) \
            .addBands(temp.rename("temp")) \
            .addBands(precip.rename("precip")) \
            .addBands(flood_occur.rename("flood_occur")) \
            .addBands(flood_duration_mean.rename("flood_duration")) \
            .addBands(flood_depth_mean.rename("flood_mean")) \
            .addBands(flood_depth_max.rename("flood_max")) \
            .addBands(terrain.select("slope").rename("slope")) \
            .addBands(terrain.select("elevation").rename("elevation"))

        # Reduce each image at the point
        sampled = combined.reduceRegion(
            reducer=reducers,
            geometry=point.geometry(),
            scale=10,
            maxPixels=1e13
        )

        # Return the point with added properties (some may be null)
        return point.set(sampled)
    
    return extract_bands_to_feature


# Extract Feature Data Function
def extract_gee_values(df, gdf_points, gee_data):
    fc_points = create_feature_collection(gdf_points)

    # Apply to all points
    extractor = make_extractor(gee_data)
    sampled = fc_points.map(extractor)

    results = sampled.getInfo()

    extracted = []
    for f in results['features']:
        props = f['properties']
        extracted.append({
            'id': props['id'],
            'land_cover_code': props.get('Map'),  # land cover code
            'slope': props.get('slope'), # Slope (degrees)
            'elevation': props.get('elevation'), # Elevation (m)
            'solar': props.get('solar') if props.get('solar') > 5 else None, # Monthly Surface Solar Radiation (J/m²)
            'temp': props.get('temp') - 273.15 if props.get('temp') is not None else None, # Mean 2m Temperature (K)
            'precip': props.get('precip'), # Mean Monthly Precipitation (m)
            'flood_occur': props.get('flood_occur'), # Annual Flood Occurrence
            'flood_duration': props.get('flood_duration'), # Mean Flood Duration (days)
            'flood_mean': props.get('flood_mean'), # Mean Flood Depth (m)
            'flood_max': props.get('flood_max'), # Max Flood Depth (m)
        })

    df.loc[:, 'id'] = df.index
    extracted = pd.DataFrame(extracted)

    # Merge the extracted data with the original DataFrame
    df_results = pd.merge(df, extracted, on='id', how='left')
    
    # Drop the 'id' column
    df_results = df_results.drop(columns=['id'])

    # Decoding land cover codes
    df_results['land_cover'] = df_results['land_cover_code'].map(LAND_LABELS)

    # drop land cover code
    df_results = df_results.drop(columns=['land_cover_code'])

    return df_results

