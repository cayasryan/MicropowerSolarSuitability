from solmate_app.assess.features import extract_features
from solmate_app.assess.rules import classify


def assess(df_coords, static_layers, gee_layers):
    df = extract_features(df_coords, static_layers, gee_layers)       # ← get features
    df = classify(df)                                                 # ← get Suitability & Remarks
    return df
