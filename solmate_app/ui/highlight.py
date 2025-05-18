import pandas as pd

from solmate_app.constants import UNSUITABLE_LC, HIGHLIGHT_COLORS, RENAME_MAPPING, THRESH


def highlight_suitability_and_features(row):
    highlight = [''] * len(row)
    columns = list(row.index)

    def col(label):
        return RENAME_MAPPING.get(label, label)

    # Suitability and remarks coloring
    suitability_val = row.get(col('suitability'))
    if suitability_val == 'Suitable':
        highlight[columns.index(col('suitability'))] = HIGHLIGHT_COLORS['suitable']
        highlight[columns.index(col('remarks'))] = HIGHLIGHT_COLORS['recommendation_suitable']
    elif suitability_val == 'Likely Unsuitable':
        highlight[columns.index(col('suitability'))] = HIGHLIGHT_COLORS['likely_unsuitable']
        highlight[columns.index(col('remarks'))] = HIGHLIGHT_COLORS['recommendation_unsuitable']

    # Feature-based highlights
    checks = [
        (col('in_protected_area'), lambda v: v == 'Yes'),
        (col('flood'), lambda v: v > THRESH['flood_extent']),
        (col('flood_mean'), lambda v: v > THRESH['flood_mean']),
        (col('flood_max'), lambda v: v > THRESH['flood_max']),
        (col('fault_line_prox'), lambda v: v < THRESH['fault_dist']),
        (col('residential_prox'), lambda v: v < THRESH['res_dist']),
        (col('solar'), lambda v: v < THRESH['solar_min']),
        (col('temp'), lambda v: v < THRESH['temp_low'] or v > THRESH['temp_high']),
        (col('precip'), lambda v: v > THRESH['precip_high']),
        (col('land_cover'), lambda v: v in UNSUITABLE_LC),
    ]

    for colname, condition in checks:
        if colname in row:
            value = row[colname]
            if pd.notnull(value):
                try:
                    if condition(value):
                        highlight[columns.index(colname)] = HIGHLIGHT_COLORS['likely_unsuitable']
                except Exception:
                    pass
                
    return highlight



def style_dataframe(df_pred, original_df):
    """
    Styles the DataFrame for display in Streamlit.
    """

    # Define the columns to be moved to the front
    cols_to_front = ['suitability', 'remarks', 'land_cover', 
                     'in_protected_area', 'protected_area_prox', 'nearest_protected_area',
                     'grid_prox', 'nearest_grid', 
                     'in_spug', 'spug_prox', 'nearest_spug',
                     'residential_prox','road_prox', 'fault_line_prox', 'airport_prox', 
    ]

    pred_cols = df_pred[cols_to_front + [col for col in df_pred.columns if col not in cols_to_front]].drop(columns=['latitude', 'longitude'])

    df_final = pd.concat([df_pred[['latitude', 'longitude']],
                        original_df.drop(columns=['latitude', 'longitude']),
                        pred_cols], axis=1)
    
    
    df_final.rename(columns=RENAME_MAPPING, inplace=True)
    styled_df = df_final.style.apply(highlight_suitability_and_features, axis=1)
    
    return df_final, styled_df

