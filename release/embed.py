"""Embed single-cell crops with the published scDINO DINOv2 model.

Needs torch, timm, tifffile and numpy -- NOT the scdino package. Everything
needed to reproduce the training-time preprocessing is read from the model's
own ``config.json``.

    # embed a labelled directory ($EVAL_DIR/<class>/*.tiff)
    python embed.py --model CSEM-AI4LS/scdino-v2-base --data $EVAL_DIR --out eval.npz

    # deterministic check of your environment (same numbers on every machine)
    python embed.py --model CSEM-AI4LS/scdino-v2-base --self-test

    # data from another imaging setup: compute new constants, then use them
    python embed.py --data $MY_DIR --calibrate my_prep.json
    python embed.py --model CSEM-AI4LS/scdino-v2-base --data $MY_DIR \
        --preprocessing my_prep.json --out my.npz

See README.md, "Preprocessing", for when to calibrate.

The output ``.npz`` holds ``embeddings`` (N, D) float32, ``paths`` (N,) and
``labels`` (N,).
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F


def preprocess(img: np.ndarray, prep: dict) -> torch.Tensor:
    """Raw ``(H, W, C)`` crop -> normalized ``(C, S, S)`` tensor.

    Exactly what scDINO training did: clip each channel at its ceiling, scale
    to [0, 1], resize, then standardize per channel. All values come from the
    model's ``config.json`` under ``scdino_preprocessing``.
    """
    ceilings = np.asarray(prep["max_vals_clip"], dtype=np.float32)
    img = np.minimum(img, ceilings).astype(np.float32) / ceilings
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    size = (int(prep["resize"]), int(prep["resize"]))
    x = F.interpolate(
        x[None], size=size, mode="bilinear", align_corners=False, antialias=True
    )[0]
    mean = torch.tensor(prep["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(prep["std"], dtype=torch.float32)[:, None, None]
    return (x - mean) / std


def load_model(model: str, device: str = "cpu"):
    """Return ``(net, prep)`` for a local directory or a Hugging Face repo id."""
    import timm

    local = (Path(model) / "config.json").is_file()
    if local:
        config = json.loads((Path(model) / "config.json").read_text())
    else:
        from huggingface_hub import hf_hub_download

        config = json.loads(Path(hf_hub_download(model, "config.json")).read_text())

    prefix = "local-dir" if local else "hf-hub"
    net = timm.create_model(f"{prefix}:{model}", pretrained=True).to(device).eval()
    return net, config["scdino_preprocessing"]


def list_crops(data: Path) -> tuple[list[str], list[str]]:
    """List the crops of a labelled directory. The class is the name.

    Two layouts are accepted:

    * ``<data>/<class>/*.tiff`` -- one directory per class;
    * ``<data>/<class>.zip`` -- one zip per class, as the public eval set is
      distributed. The crops are read straight out of the archive, so there is
      nothing to unpack. Such a crop is addressed as ``<zip>::<member>``.
    """
    paths, labels = [], []
    for archive in sorted(data.glob("*.zip")):
        with zipfile.ZipFile(archive) as handle:
            members = sorted(
                n for n in handle.namelist() if n.lower().endswith((".tif", ".tiff"))
            )
        paths += [f"{archive}::{name}" for name in members]
        labels += [archive.stem] * len(members)
    for class_dir in sorted(p for p in data.iterdir() if p.is_dir()):
        for path in sorted(class_dir.glob("*.tif*")):
            paths.append(str(path))
            labels.append(class_dir.name)
    if not paths:
        raise SystemExit(f"no .tiff crops found under {data}/<class>/ or {data}/*.zip")
    return paths, labels


def read_crop(path: str) -> np.ndarray:
    """Read one crop, from a file or from ``<zip>::<member>``."""
    if "::" not in path:
        return tifffile.imread(path)
    archive, member = path.split("::", 1)
    handle = _ARCHIVES.get(archive) or _ARCHIVES.setdefault(
        archive, zipfile.ZipFile(archive)
    )
    return tifffile.imread(io.BytesIO(handle.read(member)))


_ARCHIVES: dict[str, zipfile.ZipFile] = {}


def select(paths, labels, per_class: int, seed: int = 0):
    """A random subset of at most ``per_class`` crops per class."""
    rng = np.random.default_rng(seed)
    paths, labels = np.asarray(paths), np.asarray(labels)
    keep = np.concatenate(
        [
            rng.permutation(np.flatnonzero(labels == c))[:per_class]
            for c in np.unique(labels)
        ]
    )
    keep.sort()
    return paths[keep].tolist(), labels[keep].tolist()


def calibrate(imgs, labels=None) -> dict:
    """New clip ceilings and mean/std for data from another imaging setup.

    ``imgs`` is a representative sample of raw ``(H, W, C)`` crops. Each clip
    ceiling is the 99th percentile of its channel. With labels, it is the 99th
    percentile per class and then the highest one, as for the model's own
    values: marker channels are bright in only some classes, and pooling all
    crops gives lower ceilings for them. The mean/std are then computed on the
    clipped and scaled crops. Returns the keys that replace those in
    ``scdino_preprocessing``.
    """
    imgs = np.asarray(imgs, dtype=np.float32)
    if labels is not None and len(set(labels)) > 1:
        labels = np.asarray(labels)
        ceilings = np.max(
            [
                np.percentile(imgs[labels == c], 99, axis=(0, 1, 2))
                for c in np.unique(labels)
            ],
            axis=0,
        )
    else:
        ceilings = np.percentile(imgs, 99, axis=(0, 1, 2))
    scaled = np.minimum(imgs, ceilings) / ceilings
    mean, std = scaled.mean(axis=(0, 1, 2)), scaled.std(axis=(0, 1, 2))
    return {
        "max_vals_clip": [round(float(v), 2) for v in ceilings],
        "mean": [round(float(v), 4) for v in mean],
        "std": [round(float(v), 4) for v in std],
    }


@torch.no_grad()
def embed_paths(net, prep, paths, device="cpu", batch_size=256) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch_size):
        batch = [preprocess(read_crop(p), prep) for p in paths[i : i + batch_size]]
        out.append(net(torch.stack(batch).to(device)).float().cpu().numpy())
        print(f"\r  {min(i + batch_size, len(paths))}/{len(paths)} crops", end="")
    print()
    return np.concatenate(out)


@torch.no_grad()
def self_test(net, prep: dict) -> None:
    """Embed a fixed synthetic crop: compare with the model card."""
    ceilings = np.asarray(prep["max_vals_clip"], dtype=np.float32)
    rng = np.random.default_rng(0)
    img = (rng.random((50, 50, len(ceilings))) * ceilings).astype(np.float32)
    emb = net(preprocess(img, prep)[None])[0].float().numpy()
    print(f"embedding dim  {emb.shape[0]}")
    print(f"first 8 values {np.array2string(emb[:8], precision=5, separator=', ')}")
    print(f"norm           {np.linalg.norm(emb):.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="Hub repo id or local directory")
    ap.add_argument(
        "--data", type=Path, help="directory with one subdirectory per class"
    )
    ap.add_argument("--out", type=Path, help="output .npz")
    ap.add_argument("--self-test", action="store_true", help="synthetic check, no data")
    ap.add_argument(
        "--limit-per-class", type=int, help="random subsample for a quick run"
    )
    ap.add_argument(
        "--calibrate",
        type=Path,
        help="write new constants for the crops in --data to this JSON, then stop",
    )
    ap.add_argument(
        "--preprocessing",
        type=Path,
        help="JSON from --calibrate: constants that replace the model's own",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.calibrate:
        if not args.data:
            raise SystemExit("--calibrate needs --data")
        paths, labels = select(*list_crops(args.data), args.limit_per_class or 1000)
        constants = calibrate([read_crop(p) for p in paths], labels)
        args.calibrate.write_text(json.dumps(constants, indent=2) + "\n")
        print(f"wrote {args.calibrate} from {len(paths)} crops")
        return

    if not args.model:
        raise SystemExit("--model is required unless you pass --calibrate")
    net, prep = load_model(args.model, "cpu" if args.self_test else device)
    if args.self_test:
        self_test(net, prep)
        return
    if not args.data or not args.out:
        raise SystemExit("--data and --out are required unless you pass --self-test")
    if args.preprocessing:
        prep = {**prep, **json.loads(args.preprocessing.read_text())}

    paths, labels = list_crops(args.data)
    if args.limit_per_class:
        paths, labels = select(paths, labels, args.limit_per_class)

    print(f"{len(paths)} crops, {len(set(labels))} classes, device={device}")
    feats = embed_paths(net, prep, paths, device, args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        embeddings=feats.astype(np.float32),
        paths=np.array(paths),
        labels=np.array(labels),
        model=args.model,
    )
    print(f"wrote {args.out} -- embeddings {feats.shape}")


if __name__ == "__main__":
    main()
