"""
k-NN evaluation on pre-extracted feature CSVs.

Reads feature CSVs (e.g. from cellpose_features.py), infers labels from the
image path hierarchy, and reports top-k accuracy.

Two evaluation protocols are supported:

*Held-out split* (preferred). Separate CSVs for the train and validation
directories, matching how the DINO models are evaluated, so the two sets of
numbers are directly comparable:

    python -m scdino.classic.run_knn \\
        --train-csv features_train.csv --val-csv features_val.csv

*Random split* of a single CSV. Cheaper, but crops from the same image or well
can land on both sides of the split, which inflates accuracy relative to the
held-out protocol. Do not compare these numbers against the DINO models:

    python -m scdino.classic.run_knn features.csv --train-fraction 0.8 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scdino.eval.knn import compute_knn_accuracy, knn_classifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="k-NN classification on Cellpose feature CSVs",
    )
    parser.add_argument(
        "features_csv",
        nargs="?",
        default=None,
        help=(
            "Feature CSV to split randomly. Mutually exclusive with "
            "--train-csv/--val-csv; prefer those for numbers you intend to "
            "compare against the DINO models"
        ),
    )
    parser.add_argument(
        "--train-csv",
        default=None,
        help="Feature CSV for the training split (use with --val-csv)",
    )
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Feature CSV for the held-out validation split (use with --train-csv)",
    )
    parser.add_argument(
        "--label-column",
        default="ImagePath",
        help=(
            "Column used to derive labels.  By default the parent directory "
            "name is extracted from ImagePath (default: ImagePath)"
        ),
    )
    parser.add_argument(
        "--feature-start",
        type=int,
        default=3,
        help="First column index that contains features (default: 3)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of data used for training (default: 0.8)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=20,
        help="Number of nearest neighbours (default: 20)",
    )
    parser.add_argument(
        "-T",
        type=float,
        default=0.07,
        help="Temperature for softmax weighting (default: 0.07)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Top-k values to report (default: 1 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the train/test split (default: non-deterministic)",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=1024,
        help="Query mini-batch size for the k-NN search (default: 1024)",
    )
    parser.add_argument(
        "--train-chunk-size",
        type=int,
        default=50_000,
        help="Train-set chunk size for memory-efficient search (default: 50000)",
    )
    return parser.parse_args(argv)


def _label_from_path(image_path) -> str:
    """Class name = the directory the crop sits in, as DatasetFolder sees it."""
    return Path(str(image_path)).parent.name


def _read_features(path: str, feature_start: int, label_column: str):
    table = pd.read_csv(path)
    if label_column not in table.columns:
        raise SystemExit(
            f"{path}: no '{label_column}' column (use --label-column). "
            f"Found: {list(table.columns[:6])}..."
        )
    features = table.iloc[:, feature_start:].values
    return features, table[label_column].apply(_label_from_path)


def _run_held_out(args) -> None:
    """Evaluate a held-out validation CSV against a training CSV."""
    train_x, train_names = _read_features(
        args.train_csv, args.feature_start, args.label_column
    )
    val_x, val_names = _read_features(
        args.val_csv, args.feature_start, args.label_column
    )

    if train_x.shape[1] != val_x.shape[1]:
        raise SystemExit(
            f"feature-count mismatch: {args.train_csv} has {train_x.shape[1]} "
            f"columns, {args.val_csv} has {val_x.shape[1]}"
        )

    # Factorise jointly so both splits share one class index space; factorising
    # separately would silently permute the labels between them.
    all_labels, classes = pd.factorize(pd.concat([train_names, val_names]))
    train_labels = all_labels[: len(train_names)]
    val_labels = all_labels[len(train_names) :]

    unseen = sorted(set(val_names) - set(train_names))
    if unseen:
        print(f"WARNING: classes only in the validation set: {unseen}")

    print(
        f"Loaded {len(train_x)} train / {len(val_x)} held-out samples, "
        f"{len(classes)} classes, {train_x.shape[1]} features"
    )
    _report(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_labels).long(),
        torch.from_numpy(val_x).float(),
        torch.from_numpy(val_labels).long(),
        len(classes),
        args,
    )


def _report(train_features, train_labels, test_features, test_labels, n_classes, args):
    probs, _ = knn_classifier(
        train_features,
        train_labels,
        test_features,
        k=args.k,
        T=args.T,
        num_classes=n_classes,
        query_batch_size=args.query_batch_size,
        train_chunk_size=args.train_chunk_size,
    )
    results = compute_knn_accuracy(probs, test_labels, topk=tuple(args.topk))
    for key, value in results.items():
        print(f"kNN {key} accuracy: {value:.2f}%")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    paired = bool(args.train_csv) or bool(args.val_csv)
    if paired:
        if not (args.train_csv and args.val_csv):
            raise SystemExit("--train-csv and --val-csv must be given together")
        if args.features_csv:
            raise SystemExit(
                "give either a single CSV (random split) or "
                "--train-csv/--val-csv (held-out split), not both"
            )
        _run_held_out(args)
        return

    if not args.features_csv:
        raise SystemExit(
            "nothing to evaluate: pass a features CSV, or --train-csv and --val-csv"
        )

    table = pd.read_csv(args.features_csv)
    features = table.iloc[:, args.feature_start :].values

    labels_raw = table[args.label_column].apply(_label_from_path)
    labels, classes = pd.factorize(labels_raw)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(features))
    split = int(args.train_fraction * len(indices))

    train_features = torch.from_numpy(features[indices[:split]]).float()
    train_labels = torch.from_numpy(labels[indices[:split]]).long()
    test_features = torch.from_numpy(features[indices[split:]]).float()
    test_labels = torch.from_numpy(labels[indices[split:]]).long()

    print(
        f"Loaded {len(features)} samples ({split} train / {len(features) - split} test) "
        f"by random split, {len(classes)} classes, {features.shape[1]} features"
    )
    print(
        "NOTE: a random split can place crops from the same image on both sides. "
        "Use --train-csv/--val-csv to match how the DINO models are evaluated."
    )

    _report(
        train_features, train_labels, test_features, test_labels, len(classes), args
    )


if __name__ == "__main__":
    main()
