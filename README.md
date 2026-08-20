# scDINO

Self-supervised representation learning for multi-channel microscopy images using DINO, DINOv2 and DINOv3.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
export DATA_DIR=/path/to/data
```

`DATA_DIR` must contain one directory per split, each holding one subdirectory
per class of `.tiff` crops:

```
$DATA_DIR/
  train_all_filtered/<class>/*.tiff
  val_all_filtered/<class>/*.tiff
```

The split directory names come from `configs/datamodule/chronotype.yaml`;
override them for a different layout:

```bash
python -m scdino.train.run \
  datamodule.paths.train_dir=$DATA_DIR/train datamodule.paths.val_dir=$DATA_DIR/val
```

`uv sync` installs `scdino` into the environment as an editable package, so
`import scdino` and the `python -m scdino.*` entry points work from any working
directory. Hydra resolves `configs/` relative to the package source, so the
commands below do not need to be run from the repository root.

## Training

```bash
python -m scdino.train.run
```

Override any config via [Hydra](https://hydra.cc/) CLI:

```bash
python -m scdino.train.run model=dino model.training.max_epochs=50 hardware.devices=2 logging=wandb
```

The trained teacher backbone is saved as a HuggingFace model in `hf_model/` under the output directory. A `results.json` with metrics, parameter count, timing, and hardware info is written alongside it.

## Inference

Evaluate a trained or pretrained model using kNN:

```bash
# a model you trained (the path printed at the end of a training run)
python -m scdino.inference.run \
  local_model_path=outputs/train/dinov2_chronotype/2026-05-05/15-25-29/hf_model

# a pretrained backbone, adapted to N-channel input
python -m scdino.inference.run model=pretrained/dinov2-large local_model_path=null
```

This writes `results.json` plus UMAP, confusion-matrix and attention-heatmap
figures to the run's output directory.

To export embeddings instead of evaluating them, use the feature-store entry
point, which writes a chunked Zarr array of embeddings alongside a Parquet index
of the source image paths:

```bash
python -m scdino.inference.embed \
  local_model_path=outputs/train/.../hf_model
```

## Benchmarks

Systematic experiments via Hydra `--multirun`. Available experiment configs:

| Category | Configs |
|----------|---------|
| **Scaling** | `+experiment=scaling`, sweeping `datamodule.loader.max_train_samples` |
| **Architecture** | `+experiment=architecture/{dino,dinov2,dinov3}` |
| **Model size** | `+experiment=model_size/{dino,dinov2,dinov3}_{small,base,large}` |
| **Augmentation ablations** | `+experiment=augmentation/{no_flip,no_rotation,no_rotation_flips,no_noise,no_channel_drop,no_intensity_scale,no_intensity_shift,no_gamma,no_blur,do_centercrop,minimal}` |
| **Pretrained baselines** | `+experiment=pretrained` (used with `scdino.inference.run`) |

### Train-then-evaluate pipeline

```bash
python -m scdino.benchmark.run \
  --train-args "+experiment=scaling datamodule.loader.max_train_samples=100,500,1000 seed=42,43,44" \
  --inference-args "eval.k=20"
```

### Collecting results

```bash
python -m scdino.benchmark.collect outputs/benchmark/scaling/
```

Produces `benchmark_summary.csv` (one row per run) and
`benchmark_summary_agg.csv` (mean/std across seeds).


## Classical baseline

Extract Cellpose morphological features and evaluate with kNN:

```bash
python -m scdino.classic.cellpose_features $DATA_DIR/train_all_filtered -o features_train.csv
python -m scdino.classic.cellpose_features $DATA_DIR/val_all_filtered   -o features_val.csv
python -m scdino.classic.run_knn --train-csv features_train.csv --val-csv features_val.csv
```

```bash
python -m scdino.classic.run_knn features.csv --train-fraction 0.8 --seed 42
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
```

## Project structure

```
src/scdino/
  train/          Training entry point and Lightning Trainer
  inference/      kNN evaluation, plus Zarr feature-store export (embed.py)
  benchmark/      Orchestrator and results collection
  data/           Dataset loading and multi-crop augmentations
  models/
    backbones/    DINO / DINOv2 / DINOv3 implementations
    lightning/    Lightning training modules and shared training hooks
    huggingface/  HuggingFace model export format
  eval/           kNN classifier and neighbourhood purity
  classic/        Cellpose feature extraction baseline
  utils/          Channel adaptation for pretrained 3-channel backbones
```

## Development

Install the dev dependency group and run the test suite:

```bash
uv sync
uv run pytest
```

The suite is CPU-only and covers the kNN classifier (against a brute-force
reference), the SSL objectives, the augmentation pipeline, dataset
normalization, channel adaptation, the HuggingFace export round trip, and the
composition of every shipped Hydra config.

CI (`.github/workflows/ci.yml`) runs the test suite against the committed
`uv.lock`, and separately verifies that the licence files and the vendored-code
provenance headers described below are still present.

## License

The original scDINO source code is released under the [MIT License](LICENSE),
Copyright (c) 2026 CSEM.

**This repository also redistributes third-party code that is not MIT-licensed.**
In particular, the upper region of
`src/scdino/models/backbones/dinov3.py` is vendored from Meta's DINOv3
reference implementation and is governed by the **DINOv3 License Agreement**,
which is more restrictive than MIT and carries an acceptable-use policy
(no military, weapons, nuclear or espionage use, plus trade-control and
sanctions compliance). It also requires that a copy of the Agreement travel
with any redistribution, and that you acknowledge use of the DINO materials in
any resulting publication.

Full inventory and obligations: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Verbatim licence texts: [`licenses/`](licenses/).

If you need a uniformly MIT-licensed tree, delete
`src/scdino/models/backbones/dinov3.py` along with the `dinov3` model, config
and Lightning entry points; everything else is MIT.

Pretrained weights fetched at runtime (`facebook/dinov2-*`, `facebook/dinov3-*`)
carry their own licences, separate from the code licences above.
