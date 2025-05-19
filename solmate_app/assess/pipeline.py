import pandas as pd
import streamlit as st

from solmate_app.assess.features import extract_features
from solmate_app.assess.rules import classify


def assess(df_coords, static_layers, gee_layers):
    df_final = pd.DataFrame()
    total_rows = df_coords.shape[0]
    batch_size = 1000
    total_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
    progress_bar = None

    # Initialize progress bar if there are multiple batches
    if total_batches > 1:
        progress_bar = st.progress(0, text=f"Processing {total_batches} batches...")

    for i, start_row in enumerate(range(0, total_rows, batch_size)):
        df_batch = df_coords.iloc[start_row:start_row + batch_size]

        df_features = extract_features(df_batch, static_layers, gee_layers)
        df_classified = classify(df_features)

        df_final = pd.concat([df_final, df_classified], axis=0)

        if total_batches > 1:
            progress = (i + 1) / total_batches
            progress_bar.progress(progress, text=f"Processing batch {i+1} of {total_batches}")

    if progress_bar:
        progress_bar.empty()  # Remove the progress bar after completion

    return df_final