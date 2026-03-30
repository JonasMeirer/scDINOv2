"""
Segmentation and feature extraction pipeline for 5-channel microscopy crops.

Loads 50x50 single-cell crops, segments the central cell with CellPose SAM,
refines with dilation/smoothing (watershed fallback), and extracts
morphological + per-channel intensity features.

Usage:
    python clean_segmentation.py /path/to/crops/ -o features.csv --batch-size 256
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    label,
    maximum_filter,
    watershed_ift,
)
from skimage.measure import regionprops_table
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Channel layout: APC(0), Brightfield(1), DAPI(2), GFP(3), PE(4)
# ---------------------------------------------------------------------------
CH_NAMES = ["APC", "Brightfield", "DAPI", "GFP", "PE"]
CH_DAPI = 2
CH_BRIGHTFIELD = 1
CH_PROTEIN = [0, 3, 4]  # APC, GFP, PE
CROP_CENTER = (25, 25)

# Escalating watershed parameters: (percentile_threshold, peak_size, dilation_iters)
WATERSHED_PARAMS = [(50, 3, 1), (50, 5, 1), (60, 5, 3)]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_images(
    paths: list[str], n_workers: int = 8
) -> tuple[list[np.ndarray], list[str]]:
    """Load TIFF images in parallel. Returns (images, valid_paths)."""

    def _load(p: str):
        try:
            return p, tifffile.imread(p)
        except Exception as e:
            logger.warning("Failed to load %s: %s", p, e)
            return p, None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_load, paths))

    imgs, valid = [], []
    for p, img in results:
        if img is not None:
            imgs.append(img)
            valid.append(p)
    return imgs, valid


# ---------------------------------------------------------------------------
# CellPose input composition
# ---------------------------------------------------------------------------


def to_cellpose_input(img: np.ndarray) -> np.ndarray:
    """Build 3-channel (H, W, 3) input from a 5-channel crop."""
    apc, brightfield, dapi, gfp, pe = np.split(img, 5, axis=-1)
    return apc + dapi + gfp + pe
    
    # return np.stack(
    #     [
    #         img[..., CH_DAPI],
    #         img[..., CH_BRIGHTFIELD],
    #         np.mean(img[..., CH_PROTEIN], axis=-1),
    #     ],
    #     axis=-1,
    # )


# ---------------------------------------------------------------------------
# Mask refinement
# ---------------------------------------------------------------------------


def _watershed_fallback(
    img5: np.ndarray, threshold_pct: int = 50, peak_size: int = 3
) -> np.ndarray:
    """Simple watershed on summed fluorescence channels."""
    signal = img5[..., CH_DAPI] + np.sum(img5[..., CH_PROTEIN], axis=-1)
    smoothed = gaussian_filter(signal.astype(np.float64), sigma=1)
    binary = smoothed > np.percentile(smoothed, threshold_pct)
    dt = distance_transform_edt(binary)
    markers = label(maximum_filter(dt, size=peak_size) == dt)[0]
    return watershed_ift(binary.astype(np.uint8), markers)


def refine_single_mask(
    nucleus_mask: np.ndarray, img5: np.ndarray
) -> tuple[np.ndarray, str]:
    """Select the cell at crop center, dilate, and smooth.

    Falls back to watershed with escalating parameters if CellPose
    assigned no label to the center pixel.

    Returns (mask, method) where method is one of
    "cellpose", "watershed", or "full_crop".
    """
    cy, cx = CROP_CENTER
    center_label = nucleus_mask[cy, cx]

    if center_label != 0:
        method = "cellpose"
        selected = (nucleus_mask == center_label).astype(np.uint8)
        dilated = binary_dilation(selected, iterations=5).astype(np.uint8)
    else:
        method = "watershed"
        dilated = None
        for thresh, size, iters in WATERSHED_PARAMS:
            ws = np.squeeze(_watershed_fallback(img5, thresh, size))
            lbl = ws[cy, cx]
            if lbl != 0:
                selected = (ws == lbl).astype(np.uint8)
                dilated = binary_dilation(selected, iterations=iters).astype(np.uint8)
                break
        if dilated is None:
            logger.debug("No mask found — keeping entire crop")
            return np.ones(img5.shape[:2], dtype=np.uint16), "full_crop"

    smoothed = gaussian_filter(dilated.astype(np.float32), sigma=5)
    final = (smoothed > 0.5).astype(np.uint16)

    if final[cy, cx] == 0:
        return np.ones(img5.shape[:2], dtype=np.uint16), "full_crop"
    return final, method


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features_batch(
    masks: list[np.ndarray],
    methods: list[str],
    imgs_raw: list[np.ndarray],
    paths: list[str],
) -> list[dict]:
    """Extract morphological + per-channel intensity features.

    Uses raw (unnormalized) images so intensity values are physically meaningful.
    """
    records: list[dict] = []
    for mask, method, img, path in zip(masks, methods, imgs_raw, paths):
        labeled, n = label(mask)
        if n == 0:
            continue

        props = regionprops_table(
            labeled, properties=["area", "eccentricity", "major_axis_length"]
        )
        rec = {
            "ImageName": Path(path).name,
            "Image Path": path,
            "SegmentationMethod": method,
            "Cell_Area": np.sum(props["area"]),
            "Cell_Eccentricity": np.sum(props["eccentricity"]),
            "Cell_MajorAxisLength": np.sum(props["major_axis_length"]),
        }

        for ch_idx, ch_name in enumerate(CH_NAMES):
            intensity = regionprops_table(
                labeled,
                intensity_image=img[..., ch_idx],
                properties=["mean_intensity"],
            )
            rec[f"CellMeanIntensity_{ch_name}"] = np.mean(
                intensity["mean_intensity"]
            )

        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def process_batch(
    paths: list[str],
    model: models.CellposeModel,
    io_workers: int = 8,
) -> list[dict]:
    """Full pipeline for one batch: load -> segment -> refine -> features."""
    imgs_raw, valid_paths = load_images(paths, n_workers=io_workers)
    if not imgs_raw:
        return []

    cp_inputs = [to_cellpose_input(img) for img in imgs_raw]

    nucleus_masks = model.eval(
        cp_inputs,
        batch_size=256,
        flow_threshold=0,
        diameter=None,
        resample=True,
        normalize=True,
    )[0]

    refined = [
        refine_single_mask(nmask, img)
        for nmask, img in zip(nucleus_masks, imgs_raw)
    ]
    masks = [r[0] for r in refined]
    methods = [r[1] for r in refined]

    return extract_features_batch(masks, methods, imgs_raw, valid_paths)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CellPose segmentation + feature extraction"
    )
    parser.add_argument(
        "input_dir",
        help="Directory (searched recursively) containing 5-channel TIFF crops",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <input_dir>/features.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of images per processing batch (default: 256)",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=20,
        help="Threads for parallel TIFF loading (default: 8)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Not a directory: {input_dir}")

    output_path = Path(args.output) if args.output else "features.csv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_path.with_name(f"segmentation_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    all_paths = sorted(
        str(p) for p in input_dir.rglob("*") if p.suffix.lower() in (".tif", ".tiff")
    )
    if not all_paths:
        logger.error("No .tif/.tiff files found under %s", input_dir)
        return
    logger.info("Found %d images in %s, processing in batches of %d", len(all_paths), input_dir, args.batch_size)

    model = models.CellposeModel(gpu=True)

    all_records: list[dict] = []
    n_batches = (len(all_paths) + args.batch_size - 1) // args.batch_size
    for start in tqdm(range(0, len(all_paths), args.batch_size), total=n_batches, desc="Batches"):
        batch_paths = all_paths[start : start + args.batch_size]
        try:
            records = process_batch(batch_paths, model, io_workers=args.io_workers)
            all_records.extend(records)
        except Exception:
            logger.exception("Failed batch starting at index %d", start)

    features_df = pd.DataFrame(all_records)
    features_df.to_csv(output_path, index=False)
    logger.info("Saved %d feature rows to %s", len(features_df), output_path)


if __name__ == "__main__":
    main()
