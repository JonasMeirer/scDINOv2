#!/usr/bin/env bash
# Extract Cellpose morphological features for the classical baseline.
#
#   DATA_FOLDER   directory of .tiff crops to process (default: $DATA_DIR/val)
#   OUTPUT_FOLDER where features.csv is written    (default: feature_extraction)
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the dataset root (the folder holding the train/ and val/ class directories), e.g. export DATA_DIR=/path/to/data}"

DATA_FOLDER="${DATA_FOLDER:-$DATA_DIR/val}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-feature_extraction}"

BATCH_SIZE=256
IO_WORKERS=8
FEATURE_WORKERS=16
FEATURE_CHUNK_SIZE=8

mkdir -p "$OUTPUT_FOLDER"

uv run python src/scdino/classic/cellpose_features.py "$DATA_FOLDER" \
  -o "$OUTPUT_FOLDER/features.csv" \
  --batch-size "$BATCH_SIZE" \
  --io-workers "$IO_WORKERS" \
  --feature-workers "$FEATURE_WORKERS" \
  --feature-chunk-size "$FEATURE_CHUNK_SIZE"
