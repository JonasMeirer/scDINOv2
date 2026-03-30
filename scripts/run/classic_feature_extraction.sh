#!/usr/bin/env bash

DATA_FOLDER="/mnt/SSD/Chronotype/val"
OUTPUT_FOLDER="feature_extraction"

BATCH_SIZE=256
IO_WORKERS=8
FEATURE_WORKERS=16
FEATURE_CHUNK_SIZE=8

mkdir -p "$OUTPUT_FOLDER"

python src/scdino/classic/cellpose_features.py "$DATA_FOLDER" -o "$OUTPUT_FOLDER/features.csv" \
  --batch-size "$BATCH_SIZE" \
  --io-workers "$IO_WORKERS" \
  --feature-workers "$FEATURE_WORKERS" \
  --feature-chunk-size "$FEATURE_CHUNK_SIZE"