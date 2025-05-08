#!/bin/bash

# Parent folder containing the split parquet folders
BASE_DIR="01_processed_data"

# Subfolders to process (relative to BASE_DIR)
FOLDERS=("flood_5_split_parquet" "flood_25_split_parquet" "flood_100_split_parquet")
BATCH_SIZE=100

for FOLDER in "${FOLDERS[@]}"; do
  FULL_PATH="$BASE_DIR/$FOLDER"
  echo "Processing folder: $FULL_PATH"
  cd "$FULL_PATH" || continue

  # Find untracked, non-ignored .parquet files
  tracked_files=$(git ls-files --others --exclude-standard '*.parquet')
  files=($tracked_files)
  total=${#files[@]}

  for ((i=0; i<total; i+=BATCH_SIZE)); do
      batch=("${files[@]:i:BATCH_SIZE}")
      git add "${batch[@]}"
      git commit -m "Add $FOLDER parquet files batch $((i / BATCH_SIZE + 1))"
      git push
  done

  cd - > /dev/null
done
