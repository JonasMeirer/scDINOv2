import numpy as np
from cellpose import models

import tifffile
from PIL import Image

from torchvision.datasets import DatasetFolder

# Channel order: APC (0), brightfield (1),DAPI (2), GFP (3), PE (4)
CH_DAPI = 2
CH_BRIGHTFIELD = 1
CH_PROTEIN = [0,3,4]


def to_cellpose_rgb(img5: np.ndarray) -> np.ndarray:
    """
    Cellpose SAM expects exactly 3 channels. Use:
    - DAPI: primary nuclear signal (matches nuclei-style training).
    - Brightfield: phase/texture context for boundaries and debris.
    - Mean of protein channels: single complementary stain without picking one marker.
    """
    dapi = img5[..., CH_DAPI]
    bf = img5[..., CH_BRIGHTFIELD]
    protein = np.mean(img5[..., CH_PROTEIN], axis=-1)
    return np.stack([dapi, bf, protein], axis=-1)


dataset = DatasetFolder(
    root="/mnt/SSD/Chronotype/train",
    loader=lambda x: tifffile.imread(x),
    extensions=["tiff"],
)

# get the first 8 images
imgs5 = [dataset[i][0] for i in range(8)]

# normalize each channel, then build 3-channel input for Cellpose
#imgs5_norm = [normalize_image(img) for img in imgs5]
imgs = [to_cellpose_rgb(img) for img in imgs5]

# Initialize Cellpose model for nucleus and cell segmentation
nucleus_model = models.CellposeModel(gpu=True, model_type="nuclei")

nucleus_masks = nucleus_model.eval(
    imgs,
    batch_size=8,
    flow_threshold=0,
    diameter=None,
    resample=True,
    normalize=True,
)[0]

# save the first mask to disk as a png
Image.fromarray(nucleus_masks[0] == 1).save("mask.png")

# RGB preview: PIL needs uint8 and shape (H, W, C), not (C, H, W)
Image.fromarray((np.clip(imgs[0], 0, 1) * 255).round().astype(np.uint8)).save("image.png")

import code
code.interact(local=dict(globals(), **locals()))
