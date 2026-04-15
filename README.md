# scDINO

Self-supervised representation learning for multi-channel microscopy images using DINO, DINOv2 and DINOv3.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
export DATA_DIR=/path/to/data   # folder with train/ and val/ subdirectories of .tiff class folders
```

## Training

```bash
python -m src.scdino.train.run
```

Override any config via [Hydra](https://hydra.cc/) CLI:

```bash
python -m src.scdino.train.run model=dino training.max_epochs=50 hardware.devices=2 logging=wandb
```

The trained teacher backbone is saved as a HuggingFace model in `hf_model/` under the output directory. A `results.json` with metrics, parameter count, timing, and hardware info is written alongside it.

## Inference

Evaluate a trained or pretrained model using kNN:

```bash
python -m src.scdino.inference.run local_model_path=outputs/train/.../hf_model
python -m src.scdino.inference.run model=pretrained/dinov2-large local_model_path=null
```

## Benchmarks

Systematic experiments via Hydra `--multirun`. Available experiment configs:

| Category | Configs |
|----------|---------|
| **Scaling** | `+experiment=scaling` with varying `max_train_samples` |
| **Augmentation ablations** | `+experiment=augmentation/{no_flip,no_rotation,no_noise,no_channel_drop,no_intensity_scale,no_intensity_shift,no_gamma,no_blur,minimal}` |
| **Model sizes** | `+experiment=model_size/{tiny,small,base,large}` |

### Train-then-evaluate pipeline

```bash
python -m src.scdino.benchmark.run \
  --train-args "+experiment=scaling datamodule.loader.max_train_samples=100,500,1000 seed=42,43,44" \
  --inference-args "eval.k=20"
```

### Collecting results

```bash
python -m src.scdino.benchmark.collect outputs/benchmark/scaling/
```

Produces `benchmark_summary.csv` (per-run) and `benchmark_summary_agg.csv` (mean/std across seeds).

### Parallel and SLURM execution

```bash
python -m src.scdino.train.run --multirun hydra/launcher=joblib +experiment=scaling ...
python -m src.scdino.train.run --multirun hydra/launcher=submitit_slurm +experiment=scaling ...
```

## Classical baseline

Extract Cellpose morphological features and evaluate with kNN:

```bash
python src/scdino/classic/cellpose_features.py --data-dir $DATA_DIR/train --output features_train.csv
python src/scdino/classic/cellpose_features.py --data-dir $DATA_DIR/val --output features_val.csv
python -m src.scdino.classic.run_knn --train-csv features_train.csv --val-csv features_val.csv
```

## Configuration

All configuration is managed through Hydra YAML composition under `configs/`:

```
configs/
  train.yaml / inference.yaml         # root configs
  model/                              # dinov2, dino, pretrained HF models
  datamodule/                         # dataset paths, normalization, transforms
  experiment/                         # benchmark experiment overrides
  trainer/                            # Lightning Trainer settings
  logging/                            # console, wandb, mlflow
  launcher/                           # joblib, submitit_slurm
```

## Project structure

```
src/scdino/
  train/          Training entry point and Lightning Trainer
  inference/      kNN evaluation on trained/pretrained models
  benchmark/      Orchestrator and results collection
  data/           Dataset loading and augmentations
  models/
    backbones/    DINO and DINOv2 ViT implementations (timm)
    lightning/    Lightning training modules
    huggingface/  HuggingFace model export format
  eval/           kNN classifier
  classic/        Cellpose feature extraction baseline
  utils/          Patch embedding channel adaptation
```
