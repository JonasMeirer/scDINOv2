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
