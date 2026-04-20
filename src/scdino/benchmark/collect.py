"""Aggregate benchmark results.json files into summary CSVs.

Walks a directory tree, reads every results.json, and produces:
  - benchmark_summary.csv          (one row per run)
  - benchmark_summary_agg.csv      (grouped by condition, mean/std across seeds)

Usage:
    python -m src.scdino.benchmark.collect outputs/benchmark/scaling/
    python -m src.scdino.benchmark.collect outputs/benchmark/ --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLS = ["val_knn_top1", "val_knn_top5", "val_silhouette"]

FLAT_KEYS = {
    "seed": lambda r: r.get("seed"),
    "val_knn_top1": lambda r: (r.get("metrics") or {}).get("val_knn_top1"),
    "val_knn_top5": lambda r: (r.get("metrics") or {}).get("val_knn_top5"),
    "val_silhouette": lambda r: (r.get("metrics") or {}).get("val_silhouette"),
    "num_parameters": lambda r: (r.get("model") or {}).get("num_parameters"),
    "num_train_samples": lambda r: (r.get("data") or {}).get("num_train_samples"),
    "wall_time_seconds": lambda r: (r.get("runtime") or {}).get("wall_time_seconds"),
    "gpu": lambda r: (r.get("runtime") or {}).get("gpu"),
}


def load_results(root: Path) -> list[dict]:
    """Find and parse all results.json under *root*."""
    rows = []
    for path in sorted(root.rglob("results.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue
        row = {"run_dir": str(path.parent)}
        for col, extractor in FLAT_KEYS.items():
            row[col] = extractor(data)
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
    
    df["experiment"] = df["run_dir"].apply(lambda x: "_".join(x.split("/")[-1].split("_")[:-1]))

    agg = aggregate(df, "experiment")
    if not agg.empty:
        agg_path = out_dir / "benchmark_summary_agg.csv"
        agg.to_csv(agg_path, index=False)
        print(f"\nWrote aggregated results to {agg_path}\n")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
