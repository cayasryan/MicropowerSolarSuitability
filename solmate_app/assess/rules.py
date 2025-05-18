import pandas as pd

from solmate_app.constants import UNSUITABLE_LC, THRESH

def classify(df):
    def classify_suitability_remarks(row):
        issues = []

        if row['in_protected_area'] == "Yes":
            issues.append("In Protected Area")
        
        if pd.notnull(row['flood']) and row['flood'] > THRESH['flood_extent'] \
           or pd.notnull(row['flood_mean']) and row['flood_mean'] > THRESH['flood_mean'] \
           or pd.notnull(row['flood_max']) and row['flood_max'] > THRESH['flood_max']:
            issues.append("High Flood Risk")

        if pd.notnull(row['fault_line_prox']) and row['fault_line_prox'] < THRESH['fault_dist']:
            issues.append("Near Fault Line")

        if pd.notnull(row['residential_prox']) and row['residential_prox'] < THRESH['res_dist']:
            issues.append("Near Residential Areas")

        if pd.notnull(row['solar']) and row['solar'] < THRESH['solar_min']:
            issues.append("Low Solar Radiation")

        if pd.notnull(row['temp']) and row['temp'] < THRESH['temp_low']:
            issues.append("Low Surface Temperature")

        if pd.notnull(row['temp']) and row['temp'] > THRESH['temp_high']:
            issues.append("High Surface Temperature")

        if pd.notnull(row['precip']) and row['precip'] > THRESH['precip_high']:
            issues.append("High Precipitation")

        if row['land_cover'] in UNSUITABLE_LC:
            issues.append(f"Land Cover is {row['land_cover']}")

        if issues:
            return ['Likely Unsuitable', "; ".join(issues)]
        else:
            return ['Suitable', "No major issues detected."]

    df[['suitability', 'remarks']] = df.apply(
        lambda row: pd.Series(classify_suitability_remarks(row)),
        axis=1
    )

    return df