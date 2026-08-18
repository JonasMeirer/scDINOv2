#!/usr/bin/env bash
# Train a model on $DATA_DIR. Any extra arguments are passed through to Hydra,
# e.g.  ./scripts/run/train.sh model=dino trainer.max_epochs=50
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the dataset root (the folder holding the train/ and val/ class directories), e.g. export DATA_DIR=/path/to/data}"

uv run python -m src.scdino.train.run "$@"
