"""
k-NN evaluation on pre-extracted feature CSVs.

Reads a feature CSV (e.g. from cellpose_features.py), infers labels from
the image path hierarchy, and reports top-k accuracy.

Usage:
    python run_knn.py features.csv
    python run_knn.py features.csv --k 10 --train-fraction 0.9 --seed 42
"""

import argparse

import numpy as np
import pandas as pd
import torch

from src.scdino.eval.knn import compute_knn_accuracy, knn_classifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="k-NN classification on a feature CSV",
    )
    parser.add_argument(
        "features_csv",
        help="Path to the feature CSV (features start at column index --feature-start)",
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    table = pd.read_csv(args.features_csv)
    features = table.iloc[:, args.feature_start :].values

    labels_raw = table[args.label_column].apply(lambda x: str(x).split("/")[-2])
    labels, classes = pd.factorize(labels_raw)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(features))
    split = int(args.train_fraction * len(indices))

    train_features = torch.from_numpy(features[indices[:split]]).float()
    train_labels = torch.from_numpy(labels[indices[:split]]).long()
    test_features = torch.from_numpy(features[indices[split:]]).float()
    test_labels = torch.from_numpy(labels[indices[split:]]).long()

    print(
        f"Loaded {len(features)} samples ({split} train / {len(features) - split} test), "
        f"{len(classes)} classes, {features.shape[1]} features"
    )

    probs, _ = knn_classifier(
        train_features,
        train_labels,
        test_features,
        k=args.k,
        T=args.T,
        num_classes=len(classes),
        query_batch_size=args.query_batch_size,
        train_chunk_size=args.train_chunk_size,
    )

    topk = tuple(args.topk)
    results = compute_knn_accuracy(probs, test_labels, topk=topk)

    for k in topk:
        print(f"kNN top{k} accuracy: {results[f'top{k}']:.2f}%")


if __name__ == "__main__":
    main()
