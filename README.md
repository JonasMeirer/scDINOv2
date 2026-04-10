# scDINO

Self-supervised representation learning for multi-channel microscopy images using DINO and DINOv2. Trained models are exported in HuggingFace format and evaluated via k-nearest-neighbor (kNN) classification.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Set the data directory (a folder with `train/` and `val/` subdirectories of `.tiff` class folders):

```bash
export DATA_DIR=/path/to/data
```

## Training

Train a DINOv2 model from scratch:

```bash
python -m src.scdino.train.run
```

Override any config value via Hydra CLI:

```bash
python -m src.scdino.train.run \
  model=dino \
  training.max_epochs=50 \
  hardware.devices=2 \
  max_batches=null \
  logging=wandb
```

The trained teacher backbone is saved as a HuggingFace model in `hf_model/` under the Hydra output directory. A `results.json` with metrics, parameter count, training time, and hardware info is written alongside it.

### Configuration

All configuration is managed through [Hydra](https://hydra.cc/) YAML composition:

```
configs/
  train.yaml                          # root config (composes everything below)
  inference.yaml                      # root config for evaluation
  model/
    dinov2.yaml                       # DINOv2 ViT + training hyperparameters
    dino.yaml                         # DINO v1 (ViT or ResNet)
    pretrained/                       # HuggingFace model IDs for eval
      dinov2-small.yaml, dinov2-base.yaml, dinov2-large.yaml
      dinov3-vits16.yaml, dinov3-vitb16.yaml, dinov3-vitl16.yaml
  datamodule/
    chronotype.yaml                   # dataset paths, normalization, loader settings
    transforms/
      dinov2.yaml, dino.yaml          # augmentation parameters
      pretrained/                     # resize-only transforms for HF models
  trainer/default.yaml                # Lightning Trainer settings
  logging/
    console.yaml, wandb.yaml, mlflow.yaml
```

## Inference

Evaluate a trained model (local or HuggingFace pretrained) using kNN:

```bash
# Local model from training
python -m src.scdino.inference.run \
  local_model_path=outputs/train/.../hf_model

# HuggingFace pretrained model
python -m src.scdino.inference.run \
  model=pretrained/dinov2-large \
  local_model_path=null
```

Writes `results.json` with top-1 and top-5 kNN accuracy to the output directory.

## Benchmarks

A Hydra-native benchmark infrastructure for systematic experiments. Each benchmark is defined as a YAML experiment config and run via `--multirun`.

### Scaling laws

Vary the number of training samples:

```bash
python -m src.scdino.train.run --multirun \
  +experiment=scaling \
  datamodule.loader.max_train_samples=100,500,1000,5000,10000,50000 \
  seed=42,43,44
```

Output: `outputs/benchmark/scaling/n{samples}_seed{seed}/`

### Augmentation ablations

Disable individual augmentations to measure their contribution:

```bash
# Single ablation
python -m src.scdino.train.run +experiment=augmentation/no_flip seed=42,43,44 --multirun

# All ablations
for aug in no_flip no_rotation no_noise no_channel_drop no_intensity_scale no_blur minimal; do
  python -m src.scdino.train.run +experiment=augmentation/$aug seed=42,43,44 --multirun
done
```

Available ablations: `no_flip`, `no_rotation`, `no_noise`, `no_channel_drop`, `no_intensity_scale`, `no_blur`, `minimal` (crops + normalize only).

### Model sizes

Compare different ViT architectures:

```bash
python -m src.scdino.train.run --multirun \
  +experiment=model_size/tiny,model_size/small,model_size/base,model_size/large \
  seed=42,43,44
```

| Config | embed_dim | depth | num_heads |
|--------|-----------|-------|-----------|
| tiny   | 128       | 6     | 4         |
| small  | 256       | 12    | 8         |
| base   | 384       | 12    | 12        |
| large  | 512       | 16    | 16        |

### Train-then-evaluate pipeline

The benchmark orchestrator chains training and inference:

```bash
# Train models, then evaluate each with kNN
python -m src.scdino.benchmark.run \
  --train-args "+experiment=scaling datamodule.loader.max_train_samples=100,500,1000,5000 seed=42,43,44" \
  --inference-args "eval.k=20"

# Re-evaluate previously trained models (skip training)
python -m src.scdino.benchmark.run \
  --skip-training \
  --train-output-dir outputs/benchmark/scaling/ \
  --inference-args "eval.k=50"
```

### Collecting results

Aggregate all `results.json` files into summary CSVs:

```bash
python -m src.scdino.benchmark.collect outputs/benchmark/scaling/
```

Produces:
- `benchmark_summary.csv` -- one row per run
- `benchmark_summary_agg.csv` -- grouped by condition with mean/std across seeds

### Parallel and SLURM execution

Run sweeps in parallel on a single machine or submit to a SLURM cluster:

```bash
# 2 parallel jobs locally
python -m src.scdino.train.run --multirun \
  hydra/launcher=joblib \
  +experiment=scaling \
  datamodule.loader.max_train_samples=100,500,1000,5000

# SLURM cluster
python -m src.scdino.train.run --multirun \
  hydra/launcher=submitit_slurm \
  +experiment=scaling \
  datamodule.loader.max_train_samples=100,500,1000,5000
```

SLURM parameters (partition, GPUs, memory, timeout) are configured in `configs/launcher/submitit_slurm.yaml`.

## Classical baseline

Extract Cellpose morphological features and evaluate with kNN:

```bash
# Feature extraction
python src/scdino/classic/cellpose_features.py --data-dir $DATA_DIR/train --output features_train.csv
python src/scdino/classic/cellpose_features.py --data-dir $DATA_DIR/val --output features_val.csv

# kNN evaluation
python -m src.scdino.classic.run_knn --train-csv features_train.csv --val-csv features_val.csv
```

## Project structure

```
scDINO/
  configs/                            # Hydra YAML configuration
    experiment/                       # Benchmark experiment overrides
    launcher/                         # SLURM and parallel launcher configs
    model/, datamodule/, trainer/, logging/
  src/scdino/
    train/                            # Training entry point and Lightning Trainer
    inference/                        # kNN evaluation on trained/pretrained models
    benchmark/                        # Orchestrator and results collection
    data/                             # Dataset loading and augmentations
    models/
      backbones/                      # DINO and DINOv2 ViT implementations (timm)
      lightning/                      # Lightning training modules
      huggingface/                    # HuggingFace model export format
    eval/                             # kNN classifier
    classic/                          # Cellpose feature extraction baseline
    utils/                            # Patch embedding channel adaptation
  scripts/run/                        # Shell script wrappers
```
