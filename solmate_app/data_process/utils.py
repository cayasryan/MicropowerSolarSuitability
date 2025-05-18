

def extract_coords(df):
    # Create a lowercase-to-original mapping
    col_map = {col.lower(): col for col in df.columns}

    # Rename if needed
    if 'latitude' in col_map and 'longitude' in col_map:
        df = df.rename(columns={
            col_map['latitude']: 'latitude',
            col_map['longitude']: 'longitude'
        })

    original_df = df.copy()

    if 'latitude' in df.columns and 'longitude' in df.columns:
        return original_df, df[['latitude', 'longitude']], True
    else:
        return original_df, None, False
