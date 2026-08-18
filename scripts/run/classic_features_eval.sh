#!/usr/bin/env bash
# kNN evaluation of the Cellpose feature baseline produced by
# classic_feature_extraction.sh.
#
#   INPUT_FOLDER  directory holding features.csv (default: feature_extraction)
set -euo pipefail

INPUT_FOLDER="${INPUT_FOLDER:-feature_extraction}"

uv run python -m scdino.classic.run_knn "$INPUT_FOLDER/features.csv" \
  --train-fraction 0.8 --seed 42
