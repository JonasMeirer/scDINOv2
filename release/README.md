# Using the published scDINO weights

The published model,
[`CSEM-AI4LS/scdino-v2-base`](https://huggingface.co/CSEM-AI4LS/scdino-v2-base),
is a plain **timm** checkpoint, so **you install nothing from this
repository**. It takes 5-channel single-cell microscopy crops and returns one
128-dimensional embedding per crop.

```bash
pip install -r requirements-user.txt
```

**Start with the tutorial, [`tutorial.ipynb`](tutorial.ipynb).** It loads the
model, preprocesses the public PBMC data, classifies the cell types with a linear head
and shows the confusion matrix. It also shows how to choose the preprocessing
constants for your own data.

## Load the model and embed one crop

```python
import json
import numpy as np, tifffile, timm, torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPO = "CSEM-AI4LS/scdino-v2-base"
model = timm.create_model(f"hf-hub:{REPO}", pretrained=True).eval()
prep = json.load(open(hf_hub_download(REPO, "config.json")))["scdino_preprocessing"]


def preprocess(img, prep):
    """Raw (H, W, 5) crop -> normalized (5, 56, 56) tensor, as in training."""
    ceilings = np.asarray(prep["max_vals_clip"], dtype=np.float32)
    img = np.minimum(img, ceilings).astype(np.float32) / ceilings
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    x = F.interpolate(x[None], (prep["resize"],) * 2, mode="bilinear",
                      align_corners=False, antialias=True)[0]
    mean = torch.tensor(prep["mean"])[:, None, None]
    std = torch.tensor(prep["std"])[:, None, None]
    return (x - mean) / std


with torch.no_grad():
    embedding = model(preprocess(tifffile.imread("cell.tiff"), prep)[None])  # (1, 128)
```

`embed.py` in this folder is the same code as a script.

## Input

Crops of one cell, `H x W x 5` (50 x 50 px in training), raw camera counts, in
this channel order:

| # | Channel | Markers |
|---|---|---|
| 1 | 647 nm | CD3 (APC), CD14 (Alexa Fluor 647) |
| 2 | Brightfield | – |
| 3 | DAPI | – |
| 4 | 488 nm | CD4, CD19 (FITC) |
| 5 | 594 nm | CD8, CD16, CD56 (PE), CD20 (Alexa Fluor 594) |

## Preprocessing

Three steps turn a raw crop into the model input:

1. **Clip and scale:** clip each channel at its *ceiling* and divide by it, so
   it lies in [0, 1].
2. **Resize** to 56 x 56 (bilinear, with antialiasing).
3. **Standardize** each channel with a mean and std.

The ceilings, mean and std are statistics of the training data. They are stored
in the model's `config.json` under `scdino_preprocessing`, and the weights were
trained on inputs normalized with exactly these values.

**These constants are ideally fixed.** For data from the training setup (the
public PBMC data, or the same microscope, staining and acquisition settings),
use them unchanged.

**Other imaging setups** can give very different raw intensities: another camera
or bit depth (8-bit instead of 16-bit), other exposure times, light sources or
stains. With the model's constants, such data can be clipped almost completely,
or fill only a small part of [0, 1]. The model can perform poorly in such scenarios.

**We do not know yet how to best adapt the model to such data.** If it performs
poorly, adapting the constants is a promising first thing to try:

1. **Start with the clip ceilings**, to get a good normalization of your data to
   [0, 1]. `--calibrate` sets each ceiling to the 99th percentile of its
   channel. With labels (one directory or zip per class), it takes the 99th
   percentile per class and then the highest one, as for the model's own
   values. This matters for marker channels, which are bright in only some cell
   types: over all crops together, the percentile is lower for them.
2. **Then also the mean and std** may be worth a try, recomputed after the new
   clip and scale.

```bash
python embed.py --data $MY_DIR --calibrate my_prep.json      # new constants
python embed.py --model CSEM-AI4LS/scdino-v2-base --data $MY_DIR \
  --preprocessing my_prep.json --out out/my.npz              # use them
```

Use a representative sample (a few thousand crops, all cell types present, same
channel order), and compare the variants on labelled data from your setup, for
example with a linear head as in the tutorial. Keep what works; experimentation
may be necessary.

A simulated check on the public data supports this order. It changes only the
intensities of the same images, so it is not a test on another microscope. Mean
cosine similarity to the embeddings of the unchanged crops (1 = identical):

| Simulated change | Model constants | New ceilings only | New mean/std only | Both new |
|---|---|---|---|---|
| none | **1.00** | 0.98 | 0.98 | 0.97 |
| other gain per channel | 0.58 | **0.98** | 0.89 | 0.97 |
| other gain + offset | 0.28 | 0.47 | 0.82 | **0.95** |

Recomputing normalization statistics on new data is a known, simple remedy for
domain shift with natural images, but it has not
been tested for this model on other datasets. New constants also cannot correct
differences in resolution, optics or staining patterns. For a very different
setup, the model may need retraining.

## Embed a directory of crops

Expected layout, one directory per class (or one directory for unlabelled data):

```
$EVAL_DIR/
  <class_a>/*.tiff
  <class_b>/*.tiff
```

One zip per class also works, which is how the public eval set is distributed.
The crops are read out of the archives, so there is nothing to unpack:

```
$EVAL_DIR/
  <class_a>.zip      # holds <class_a>/*.tiff
  <class_b>.zip
```

```bash
# CPU works; the GPU is used automatically when available
python embed.py --model CSEM-AI4LS/scdino-v2-base --data $EVAL_DIR --out out/eval.npz
```

The output `.npz` holds `embeddings` (N, 128), `paths` (N,) and `labels` (N,),
ready for clustering, a kNN or a linear probe. The tutorial shows a linear head with a
confusion matrix.

Useful flags: `--limit-per-class 500` for a quick run (random crops per class),
`--batch-size`, `--device`.
`python embed.py --model CSEM-AI4LS/scdino-v2-base --self-test` embeds a fixed
synthetic crop, to compare two environments.

## Files

| File | Purpose |
|---|---|
| `tutorial.ipynb` | walkthrough: load, preprocess, embed, classify, own data |
| `embed.py` | load the model, embed a directory of crops, `--calibrate`, `--self-test` |
| `requirements-user.txt` | what to install |

The model is not a diagnostic tool and not for clinical use.
