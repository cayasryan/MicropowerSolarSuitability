import pandas as pd
import streamlit as st
import datetime as dt
import time

from solmate_app.assess.features import extract_features
from solmate_app.assess.rules import classify
from solmate_app.config import BATCH_SIZE


def assess(df_coords, static_layers, gee_layers):
    df_final = pd.DataFrame()
    total_rows = df_coords.shape[0]
    batch_size = BATCH_SIZE
    total_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
    progress_bar = None

    # Initialize progress bar if there are multiple batches
    if total_batches > 1:
        progress_bar = st.progress(0, text=f"Processing {total_batches} batches...")
        eta_text = st.empty()

    start_time = time.time()

    for i, start_row in enumerate(range(0, total_rows, batch_size)):
        df_batch = df_coords.iloc[start_row:start_row + batch_size]

        df_features = extract_features(df_batch, static_layers, gee_layers)
        df_classified = classify(df_features)

        df_final = pd.concat([df_final, df_classified.reset_index(drop=True)], axis=0)

        if total_batches > 1:
            progress = (i + 1) / total_batches
            progress_bar.progress(progress, text=f"Processing batch {i+1} of {total_batches}")

            # Update ETA text
            elapsed = time.time() - start_time
            avg_batch_time = elapsed / (i + 1)
            remaining_batches = total_batches - (i + 1)
            eta_seconds = avg_batch_time * remaining_batches
            eta_str = str(dt.timedelta(seconds=int(eta_seconds)))
            eta_text.text(f"Estimated time remaining: {eta_str}")

    if total_batches > 1:
        progress_bar.empty()
        eta_text.empty()  # Remove the progress bar after completion

    return df_final