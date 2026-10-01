#!/bin/bash

# Default values
LIMIT=1000
MIN_ROUNDS=5
START_ROUND=3

# Parse command line arguments.
while getopts "l:m:s:" opt; do
  case $opt in
    l)
      LIMIT="$OPTARG"
      ;;
    m)
      MIN_ROUNDS="$OPTARG"
      ;;
    s)
      START_ROUND="$OPTARG"
      ;;
    *)
      echo "Usage: $0 [-l limit] [-m min_rounds] [-s start_round]"
      exit 1
      ;;
  esac
done

# Calculate round_number as start_round - 1.
ROUND_NUMBER=$((START_ROUND - 1))

# Download the JSON file.
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Run Python preprocessing scripts with the parsed parameters.
python3 data_preprocessing.py --parse 1
python3 concat_input.py --limit "$LIMIT"
python3 prepare_run_dataset.py --min_rounds "$MIN_ROUNDS" --start_round "$START_ROUND"
python3 prepare_warmup_dataset.py --min_rounds "$MIN_ROUNDS" --round_number "$ROUND_NUMBER"

# List of files to delete.
files=(
  "modified_file.json"
  "ShareGPT.json"
  "ShareGPT_V3_unfiltered_cleaned_split.json"
)

# Loop over each file and delete it if it exists.
for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    rm "$file" && echo "Deleted: $file" || echo "Failed to delete: $file"
  else
    echo "File not found: $file"
  fi
done
