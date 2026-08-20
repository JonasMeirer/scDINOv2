"""Aggregate benchmark results.json files into summary CSVs.

Walks a directory tree, reads every results.json, and produces:
  - benchmark_summary.csv          (one row per run)
  - benchmark_summary_agg.csv      (grouped by condition, mean/std across seeds)

Usage:
    python -m scdino.benchmark.collect outputs/benchmark/scaling/
    python -m scdino.benchmark.collect outputs/benchmark/ --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLS = [
    "val_knn_top1",
    "val_knn_top5",
    "val_silhouette",
    "val_purity@1",
    "val_purity@10",
    "val_purity@100",
    "val_purity@1000",
    "val_purity_umap@1",
    "val_purity_umap@10",
    "val_purity_umap@100",
    "val_purity_umap@1000",
]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Skipping {path}: {exc}")
        return None


def _stage_of(doc: dict, path: Path) -> str:
    """Which stage wrote this results.json.

    Both stages use the same filename. New files carry an explicit `stage`;
    older ones are identified by the `inference/` directory that
    benchmark.run writes them into, or by keys only training records.
    """
    stage = doc.get("stage")
    if stage in ("train", "inference"):
        return stage
    if path.parent.name == "inference":
        return "inference"
    return "train" if "config" in doc and "runtime" in doc else "inference"


def _run_dir_for(stage: str, path: Path) -> Path:
    """The run a results.json belongs to.

    Inference results live in `<run>/inference/`, so they fold back onto the
    training run one level up. Anything else is its own run.
    """
    if stage == "inference" and path.parent.name == "inference":
        return path.parent.parent
    return path.parent


def _experiment_name(run_dir: Path, seed) -> str:
    """Sweep arm name, i.e. the subdir with its `_seed<N>` suffix removed.

    Uses the seed recorded in the results file rather than splitting on "_",
    so arm names containing underscores (`no_intensity_shift`) survive.
    """
    name = run_dir.name
    if seed is not None:
        suffix = f"_seed{seed}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    stem, sep, tail = name.rpartition("_")
    return stem if sep and tail.startswith("seed") else name


def load_results(root: Path) -> list[dict]:
    """One row per run, merging the training and inference results.json.

    The two stages write the same filename in different places; keying on the
    run directory instead of the file is what stops a run appearing twice and
    what lets the inference-only metrics (purity, UMAP, HDBSCAN) reach the
    summary at all.
    """
    runs: dict[Path, dict] = {}
    for path in sorted(root.rglob("results.json")):
        doc = _read_json(path)
        if doc is None:
            continue
        stage = _stage_of(doc, path)
        run_dir = _run_dir_for(stage, path)
        runs.setdefault(run_dir, {})[stage] = doc

    rows = []
    for run_dir, docs in sorted(runs.items()):
        train_doc, infer_doc = docs.get("train", {}), docs.get("inference", {})
        # Inference wins on the keys both report: it is computed on the final
        # exported model with the full evaluation protocol, whereas training
        # reports whatever the last validation epoch produced.
        metrics = {
            **(train_doc.get("metrics") or {}),
            **(infer_doc.get("metrics") or {}),
        }
        seed = train_doc.get("seed", infer_doc.get("seed"))

        row = {
            "run_dir": str(run_dir),
            "experiment": _experiment_name(run_dir, seed),
            "seed": seed,
            "has_train": bool(train_doc),
            "has_inference": bool(infer_doc),
        }
        for col in METRIC_COLS:
            row[col] = metrics.get(col)
        rows.append(row)
    return rows


def aggregate(df: pd.DataFrame, aggregate_column: str) -> pd.DataFrame:
    """Group by condition (everything except seed and metrics) and compute stats."""

    agg_dict = {m: ["mean", "std", "count"] for m in METRIC_COLS if m in df.columns}
    agg = df.groupby(aggregate_column, dropna=False).agg(agg_dict)
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    first_count = next(
        (f"{m}_count" for m in METRIC_COLS if f"{m}_count" in agg.columns), None
    )
    if first_count:
        agg = agg.rename(columns={first_count: "n_seeds"})
        other_counts = [
            f"{m}_count" for m in METRIC_COLS if f"{m}_count" in agg.columns
        ]
        agg = agg.drop(columns=other_counts, errors="ignore")

    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect benchmark results")
    parser.add_argument("root", type=str, help="Root directory to search")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for CSVs (defaults to root)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(root)
    if not rows:
        print(f"No results.json files found under {root}")
        return

    df = pd.DataFrame(rows)
    summary_path = out_dir / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Wrote {len(df)} rows to {summary_path}\n")
    print(df.to_string(index=False))

    missing = df[~df["has_inference"]]["run_dir"].tolist()
    if missing:
        print(
            f"\n[NOTE] {len(missing)} run(s) have no inference results, so their "
            "purity/UMAP columns are empty. Run `scdino.benchmark.run` (or its "
            "inference stage) over the sweep to populate them."
        )

    # `experiment` is assigned in load_results from the recorded seed, so arm
    # names containing underscores survive.
    df = df.drop(columns=["run_dir", "has_train", "has_inference"], inplace=False)
    df = df.astype(
        {col: float for col in df.columns if col != "experiment"}, errors="ignore"
    )

    agg = aggregate(df, "experiment")
    if not agg.empty:
        agg_path = out_dir / "benchmark_summary_agg.csv"
        agg.to_csv(agg_path, index=False)
        print(f"\nWrote aggregated results to {agg_path}\n")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
