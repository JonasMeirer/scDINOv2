"""
Segmentation and full feature extraction for 5-channel microscopy crops.

Loads 50x50 single-cell crops, segments the central cell with CellPose SAM,
refines with dilation/smoothing (watershed fallback), and extracts
morphological, intensity (bulk + edge + mass displacement), and texture
(Haralick GLCM + Gabor) features per channel.

Usage:
    python clean_segmentation_full_features.py /path/to/crops/ -o features.csv
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
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label,
    maximum_filter,
    sum as ndi_sum,
    watershed_ift,
)
from skimage.feature import graycomatrix
from skimage.filters import gabor
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

WATERSHED_PARAMS = [(50, 3, 1), (50, 5, 1), (60, 5, 3)]

HARALICK_NAMES = [
    "AngularSecondMoment", "Contrast", "Correlation",
    "DifferenceEntropy", "DifferenceVariance", "Entropy",
    "InfoMeas1", "InfoMeas2", "InverseDifferenceMoment",
    "SumAverage", "SumEntropy", "SumVariance", "Variance",
]
TEXTURE_NAMES = HARALICK_NAMES + ["Gabor"]

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
            logger.debug("No mask found - keeping entire crop")
            return np.ones(img5.shape[:2], dtype=np.uint16), "full_crop"

    smoothed = gaussian_filter(dilated.astype(np.float32), sigma=5)
    final = (smoothed > 0.5).astype(np.uint16)

    if final[cy, cx] == 0:
        return np.ones(img5.shape[:2], dtype=np.uint16), "full_crop"
    return final, method


def keep_largest_object(mask: np.ndarray) -> np.ndarray | None:
    """Reduce mask to its single largest connected component."""
    mask = np.squeeze(mask).astype(bool)
    labeled_mask, n = label(mask)
    if n == 0:
        return None
    if n == 1:
        return mask.astype(np.uint8)
    sizes = ndi_sum(mask, labeled_mask, index=np.arange(1, n + 1))
    largest = np.argmax(sizes) + 1
    return (labeled_mask == largest).astype(np.uint8)


# ---------------------------------------------------------------------------
# Intensity helpers
# ---------------------------------------------------------------------------

_NAN_STATS = {
    "IntegratedIntensity": np.nan,
    "LowerQuartileIntensity": np.nan,
    "MADIntensity": np.nan,
    "MaxIntensity": np.nan,
    "MeanIntensity": np.nan,
    "MedianIntensity": np.nan,
    "MinIntensity": np.nan,
    "StdIntensity": np.nan,
    "UpperQuartileIntensity": np.nan,
}


def _safe_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return dict(_NAN_STATS)
    med = np.median(values)
    return {
        "IntegratedIntensity": np.sum(values),
        "LowerQuartileIntensity": np.percentile(values, 25),
        "MADIntensity": np.median(np.abs(values - med)),
        "MaxIntensity": np.max(values),
        "MeanIntensity": np.mean(values),
        "MedianIntensity": med,
        "MinIntensity": np.min(values),
        "StdIntensity": np.std(values),
        "UpperQuartileIntensity": np.percentile(values, 75),
    }


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    edge = m & ~binary_erosion(m)
    if edge.sum() == 0:
        edge = m
    return edge


def _mass_displacement(mask: np.ndarray, channel: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return np.nan
    geom = coords.mean(axis=0)
    intensities = channel[mask].astype(float)
    total = intensities.sum()
    if total <= 0:
        return 0.0
    weighted = np.average(coords, axis=0, weights=intensities)
    return float(np.linalg.norm(weighted - geom))


def _add_intensity_features(
    rec: dict, mask: np.ndarray, channel: np.ndarray,
    compartment: str, ch_name: str,
) -> None:
    """Append bulk + edge intensity stats and mass displacement to rec."""
    bool_mask = mask.astype(bool)
    edge = _edge_mask(bool_mask)

    for stat, val in _safe_stats(channel[bool_mask]).items():
        rec[f"{compartment}_Intensity_{stat}_{ch_name}"] = val
    for stat, val in _safe_stats(channel[edge]).items():
        rec[f"{compartment}_Intensity_{stat}Edge_{ch_name}"] = val
    rec[f"{compartment}_Intensity_MassDisplacement_{ch_name}"] = _mass_displacement(
        bool_mask, channel
    )


# ---------------------------------------------------------------------------
# Texture helpers (Haralick GLCM + Gabor)
# ---------------------------------------------------------------------------


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _haralick_from_glcm(P: np.ndarray) -> dict:
    """Compute 13 Haralick features from a single normalised 2-D GLCM."""
    eps = 1e-12
    P = P.astype(float)
    P = P / (P.sum() + eps)
    n = P.shape[0]
    i, j = np.indices((n, n))

    px = P.sum(axis=1)
    py = P.sum(axis=0)
    mux = np.sum(i[:, 0] * px)
    muy = np.sum(j[0, :] * py)
    sigx = np.sqrt(np.sum(((i[:, 0] - mux) ** 2) * px))
    sigy = np.sqrt(np.sum(((j[0, :] - muy) ** 2) * py))

    asm = np.sum(P**2)
    contrast = np.sum((i - j) ** 2 * P)
    correlation = (
        np.sum((i - mux) * (j - muy) * P) / (sigx * sigy)
        if sigx > 0 and sigy > 0
        else 0.0
    )
    idm = np.sum(P / (1.0 + (i - j) ** 2))
    ent = _entropy(P)

    mu = np.sum(i * P)
    variance = np.sum(((i - mu) ** 2) * P)

    p_xplusy = np.zeros(2 * n - 1)
    for k in range(2 * n - 1):
        p_xplusy[k] = np.sum(P[(i + j) == k])

    p_xminusy = np.zeros(n)
    for k in range(n):
        p_xminusy[k] = np.sum(P[np.abs(i - j) == k])

    sum_average = np.sum(np.arange(2 * n - 1) * p_xplusy)
    sum_entropy = _entropy(p_xplusy)
    sum_variance = np.sum(
        ((np.arange(2 * n - 1) - sum_average) ** 2) * p_xplusy
    )

    diff_entropy = _entropy(p_xminusy)
    diff_variance = (
        np.var(
            np.repeat(np.arange(n), np.round(p_xminusy * 1000).astype(int))
        )
        if np.sum(p_xminusy) > 0
        else 0.0
    )

    HX = _entropy(px)
    HY = _entropy(py)
    HXY = ent
    px_py = np.outer(px, py)
    HXY1 = -np.sum(P[P > 0] * np.log2(px_py[P > 0] + eps))
    HXY2 = _entropy(px_py)

    info1 = (HXY - HXY1) / max(HX, HY, eps)
    info2 = np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (HXY2 - HXY))))

    return {
        "AngularSecondMoment": asm,
        "Contrast": contrast,
        "Correlation": correlation,
        "DifferenceEntropy": diff_entropy,
        "DifferenceVariance": diff_variance,
        "Entropy": ent,
        "InfoMeas1": info1,
        "InfoMeas2": info2,
        "InverseDifferenceMoment": idm,
        "SumAverage": sum_average,
        "SumEntropy": sum_entropy,
        "SumVariance": sum_variance,
        "Variance": variance,
    }


def _quantize_for_texture(
    channel: np.ndarray, mask: np.ndarray, levels: int = 32
) -> np.ndarray:
    """Quantise masked intensities to [0, levels-1]; outside = 0."""
    bool_mask = mask.astype(bool)
    out = np.zeros_like(channel, dtype=np.uint8)
    vals = channel[bool_mask].astype(float)
    if vals.size == 0:
        return out
    vmin, vmax = vals.min(), vals.max()
    if vmax <= vmin:
        return out
    scaled = np.clip((channel.astype(float) - vmin) / (vmax - vmin), 0, 1)
    out[bool_mask] = np.floor(scaled * (levels - 1)).astype(np.uint8)[bool_mask]
    return out


def _add_texture_features(
    rec: dict, mask: np.ndarray, channel: np.ndarray,
    compartment: str, ch_name: str,
    distance: int = 6, levels: int = 32,
) -> None:
    """Append Haralick (GLCM, 4 angles averaged) + Gabor features to rec."""
    prefix = f"{compartment}_Texture"
    bool_mask = mask.astype(bool)

    if bool_mask.sum() < 10:
        for name in TEXTURE_NAMES:
            rec[f"{prefix}_{name}_{ch_name}_{distance}"] = np.nan
        return

    quantized = _quantize_for_texture(channel, bool_mask, levels)
    coords = np.argwhere(bool_mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    crop = quantized[r0:r1, c0:c1]

    glcm = graycomatrix(
        crop,
        distances=[distance],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )

    collected: dict[str, list] = {}
    for angle_idx in range(glcm.shape[3]):
        for k, v in _haralick_from_glcm(glcm[:, :, 0, angle_idx]).items():
            collected.setdefault(k, []).append(v)

    for k, vals in collected.items():
        rec[f"{prefix}_{k}_{ch_name}_{distance}"] = np.mean(vals)

    ch_f = channel.astype(np.float32)
    mvals = ch_f[bool_mask]
    if mvals.size == 0 or mvals.max() <= mvals.min():
        gabor_val = np.nan
    else:
        normed = (ch_f - mvals.min()) / (mvals.max() - mvals.min() + 1e-12)
        real, imag = gabor(normed, frequency=0.1)
        gabor_val = float(np.mean(np.sqrt(real**2 + imag**2)[bool_mask]))

    rec[f"{prefix}_Gabor_{ch_name}_{distance}"] = gabor_val


# ---------------------------------------------------------------------------
# Feature extraction (full)
# ---------------------------------------------------------------------------


def extract_features_batch(
    masks: list[np.ndarray],
    methods: list[str],
    imgs_raw: list[np.ndarray],
    paths: list[str],
) -> list[dict]:
    """Extract morphological + intensity + texture features per crop.

    Uses raw (unnormalized) images so intensity values are physically meaningful.
    """
    records: list[dict] = []
    for mask, method, img, path in zip(masks, methods, imgs_raw, paths):
        cell_mask = keep_largest_object(mask)
        if cell_mask is None:
            continue

        labeled, n = label(cell_mask)
        if n == 0:
            continue

        props = regionprops_table(
            labeled, properties=["area", "eccentricity", "major_axis_length"]
        )
        rec: dict = {
            "ImageName": Path(path).name,
            "ImagePath": path,
            "SegmentationMethod": method,
            "Cell_Area": np.sum(props["area"]),
            "Cell_Eccentricity": np.sum(props["eccentricity"]),
            "Cell_MajorAxisLength": np.sum(props["major_axis_length"]),
        }

        for ch_idx, ch_name in enumerate(CH_NAMES):
            ch = img[..., ch_idx]

            intensity = regionprops_table(
                labeled, intensity_image=ch, properties=["mean_intensity"]
            )
            rec[f"CellMeanIntensity_{ch_name}"] = np.mean(
                intensity["mean_intensity"]
            )

            _add_intensity_features(rec, cell_mask, ch, "Cells", ch_name)
            _add_texture_features(rec, cell_mask, ch, "Cells", ch_name)

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
        batch_size=8,
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
        description="CellPose segmentation + full feature extraction"
    )
    parser.add_argument(
        "input_dir",
        help="Directory (searched recursively) containing 5-channel TIFF crops",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: features.csv in working directory)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Images per processing batch (default: 256)",
    )
    parser.add_argument(
        "--io-workers", type=int, default=8,
        help="Threads for parallel TIFF loading (default: 8)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Not a directory: {input_dir}")

    output_path = Path(args.output) if args.output else Path("features.csv")

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
    logger.info(
        "Found %d images in %s, processing in batches of %d",
        len(all_paths), input_dir, args.batch_size,
    )

    model = models.CellposeModel(gpu=True)

    all_records: list[dict] = []
    n_batches = (len(all_paths) + args.batch_size - 1) // args.batch_size
    for start in tqdm(
        range(0, len(all_paths), args.batch_size), total=n_batches, desc="Batches"
    ):
        batch_paths = all_paths[start : start + args.batch_size]
        try:
            records = process_batch(
                batch_paths, model, io_workers=args.io_workers
            )
            all_records.extend(records)
        except Exception:
            logger.exception("Failed batch starting at index %d", start)

    features_df = pd.DataFrame(all_records)
    features_df.to_csv(output_path, index=False)
    logger.info("Saved %d feature rows to %s", len(features_df), output_path)


if __name__ == "__main__":
    main()
