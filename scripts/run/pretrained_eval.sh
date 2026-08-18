#!/usr/bin/env bash
# Evaluate a pretrained or locally trained model with kNN. Extra arguments are
# passed through to Hydra, e.g.
#   ./scripts/run/pretrained_eval.sh model=pretrained/dinov2-large
#   ./scripts/run/pretrained_eval.sh local_model_path=outputs/train/.../hf_model
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the dataset root (the folder holding the train/ and val/ class directories), e.g. export DATA_DIR=/path/to/data}"

uv run python -m src.scdino.inference.run "$@"
