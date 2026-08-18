import json
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import umap
from omegaconf import DictConfig
from sklearn.metrics import silhouette_score

from scdino.eval.knn import knn_classifier, compute_knn_accuracy
from scdino.eval.purity import compute_purity, purity_per_class
from scdino.inference.utils import (
    build_model,
    extract_features,
    plot_confusion_matrix,
    plot_umap_by_class,
    plot_umap_correctness,
    run_hdbscan,
    sample_per_class,
    sync_with_training_config,
)


@hydra.main(config_path="../../../configs", config_name="inference.yaml")
def run_inference(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    hydra_out = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    sync_with_training_config(cfg)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("predict")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader(shuffle=True)

    max_train_batches = (
        cfg.max_train_batches
        if cfg.max_train_batches is not None
        else len(train_loader)
    )
    max_val_batches = (
        cfg.max_val_batches if cfg.max_val_batches is not None else len(val_loader)
    )
    n_cells_per_class = cfg.get("n_cells_per_class")
    assert n_cells_per_class > 0, "n_cells_per_class must be greater than 0"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.datamodule.loader.accelerator == "gpu"
        else "cpu"
    )
    model_id = cfg.local_model_path if cfg.get("local_model_path") else cfg.model.name
    flavor = cfg.get("channel_adaptation_flavor", "mean")
    num_channels = cfg.datamodule.loader.num_channels

    embed, attn_fn = build_model(model_id, flavor, num_channels, device)

    ext = cfg.eval.get("plot_format", "png").lower()

    train_features, train_labels, train_n, train_t = extract_features(
        train_loader,
        embed,
        device,
        max_train_batches,
        attn_fn=attn_fn,
        attn_path=hydra_out / f"attn_heatmap.{ext}",
    )
    test_features, test_labels, test_n, test_t = extract_features(
        val_loader,
        embed,
        device,
        max_val_batches,
    )
    infer_samples = train_n + test_n
    infer_seconds = train_t + test_t
    throughput_samples_per_min = (
        (infer_samples / infer_seconds) * 60.0 if infer_seconds > 0 else float("nan")
    )
    print(
        f"Inference throughput: {throughput_samples_per_min:.2f} samples/min "
        f"({infer_samples} samples in {infer_seconds:.2f}s)"
    )

    num_classes = len(torch.unique(train_labels))
    print(f"Number of classes: {num_classes}")

    probs, _ = knn_classifier(
        train_features,
        train_labels,
        test_features,
        k=cfg.eval.k,
        T=cfg.eval.T,
        num_classes=num_classes,
        query_batch_size=cfg.eval.query_batch_size,
        train_chunk_size=cfg.eval.train_chunk_size,
    )
    results = compute_knn_accuracy(probs, test_labels, topk=(1, 5))
    print(f"kNN top1 accuracy: {results['top1']:.2f}%")
    if "top5" in results:
        print(f"kNN top5 accuracy: {results['top5']:.2f}%")  # only if available

    test_features_np = test_features.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    sample_size = min(10_000, len(test_features_np))
    results["silhouette"] = silhouette_score(
        test_features_np,
        test_labels_np,
        sample_size=sample_size,
        random_state=42,
    )
    print(f"Silhouette score: {results['silhouette']:.4f}")

    sampled_orig_idx = sample_per_class(test_labels_np, n_cells_per_class)
    test_features_np_sampled = test_features_np[sampled_orig_idx]
    test_labels_np_sampled = test_labels_np[sampled_orig_idx]

    preds_np = probs.argmax(dim=1).cpu().numpy()
    correct_sampled = preds_np[sampled_orig_idx] == test_labels_np_sampled

    umap_cfg = cfg.eval.umap
    reducer = umap.UMAP(
        n_components=umap_cfg.n_components,
        n_neighbors=umap_cfg.n_neighbors,
        min_dist=umap_cfg.min_dist,
        metric=umap_cfg.metric,
    )
    umap_features = reducer.fit_transform(test_features_np_sampled)
    results["silhouette_umap"] = silhouette_score(
        umap_features,
        test_labels_np_sampled,
        random_state=42,
    )
    print(f"Silhouette score (UMAP): {results['silhouette_umap']:.4f}")

    y = test_labels_np_sampled
    y_class = (
        pd.Series(y)
        .map({val: key for key, val in datamodule.val_dataset.class_to_idx.items()})
        .values
    )

    ks = (1, 10, 100, 1000)
    same_label, purities = compute_purity(test_features_np_sampled, y, ks=ks)
    same_label_umap, purities_umap = compute_purity(umap_features, y, ks=ks)
    for k in ks:
        results[f"purity@{k}"] = purities[k]
        results[f"purity_umap@{k}"] = purities_umap[k]
        print(
            f"Purity@{k}: {purities[k]:.3f}  |  Purity_umap@{k}: {purities_umap[k]:.3f}"
        )

    print("Purity@100 per class:")
    per_class = purity_per_class(same_label, y_class, k=100)
    per_class_umap = purity_per_class(same_label_umap, y_class, k=100)
    results["val_purity@100_classes"] = {
        f"class_{cls}": format(v, ".4f") for cls, v in per_class.items()
    }
    for cls, v in per_class.items():
        print("\t", cls, format(v, ".4f"))
    results["val_purity_umap@100_classes"] = {
        f"class_{cls}": format(v, ".4f") for cls, v in per_class_umap.items()
    }
    for cls, v in per_class_umap.items():
        print("\t", cls, format(v, ".4f"))

    n_per_class = cfg.eval.viz.cells_per_class
    class_names = getattr(datamodule.val_dataset, "classes", None)
    rng = np.random.default_rng(cfg.seed)
    viz_idx = sample_per_class(test_labels_np_sampled, n_per_class, rng=rng)
    viz_pts = umap_features[viz_idx, :2]
    viz_lab = test_labels_np_sampled[viz_idx]
    viz_correct = correct_sampled[viz_idx]

    umap_path = hydra_out / f"umap.{ext}"
    umap_correct_path = hydra_out / f"umap_correctness.{ext}"
    plot_umap_by_class(viz_pts, viz_lab, n_per_class, class_names, umap_path)
    print(f"Saved UMAP visualization to {umap_path}")
    plot_umap_correctness(viz_pts, viz_correct, n_per_class, umap_correct_path)
    print(f"Saved UMAP correctness visualization to {umap_correct_path}")

    cm_cfg = cfg.eval.get("confusion_matrix")
    if cm_cfg is not None and cm_cfg.get("enabled", False):
        class_names = getattr(datamodule.val_dataset, "classes", None)
        cm_path = hydra_out / f"confusion_matrix.{ext}"
        plot_confusion_matrix(
            test_labels_np,
            preds_np,
            class_names,
            cm_path,
            normalize=cm_cfg.get("normalize", True),
        )
        print(f"Saved confusion matrix to {cm_path}")

    hdbscan_cfg = cfg.eval.get("hdbscan")
    if hdbscan_cfg is not None and hdbscan_cfg.get("enabled", False):
        plot_path = (
            hydra_out / f"hdbscan_cluster_distance_matrix.{ext}"
            if hdbscan_cfg.get("plot_distance_matrix", False)
            else None
        )
        hdb = run_hdbscan(test_features_np_sampled, hdbscan_cfg, plot_path=plot_path)
        results["hdbscan_n_clusters"] = hdb["n_clusters"]
        results["hdbscan_silhouette"] = hdb["silhouette"]

    metrics_out = {
        "val_knn_top1": format(results["top1"], ".4f"),
        # "val_knn_top5": format(results['top5'], ".4f"),
        "val_silhouette": format(results["silhouette"], ".4f"),
        "val_silhouette_umap": format(results["silhouette_umap"], ".4f"),
        "val_purity@1": format(results["purity@1"], ".4f"),
        "val_purity@10": format(results["purity@10"], ".4f"),
        "val_purity@100": format(results["purity@100"], ".4f"),
        "val_purity@1000": format(results["purity@1000"], ".4f"),
        "val_purity_umap@1": format(results["purity_umap@1"], ".4f"),
        "val_purity_umap@10": format(results["purity_umap@10"], ".4f"),
        "val_purity_umap@100": format(results["purity_umap@100"], ".4f"),
        "val_purity_umap@1000": format(results["purity_umap@1000"], ".4f"),
        "val_purity@100_classes": results["val_purity@100_classes"],
        "val_purity_umap@100_classes": results["val_purity_umap@100_classes"],
        "inference_throughput_samples_per_min": format(
            throughput_samples_per_min, ".2f"
        ),
    }
    if "top5" in results:  # only if available
        metrics_out["val_knn_top5"] = format(results["top5"], ".4f")

    if "hdbscan_n_clusters" in results:
        metrics_out["val_hdbscan_n_clusters"] = results["hdbscan_n_clusters"]
        metrics_out["val_hdbscan_silhouette"] = format(
            results["hdbscan_silhouette"], ".4f"
        )

    with open(hydra_out / "results.json", "w") as f:
        json.dump({"seed": cfg.seed, "metrics": metrics_out}, f)


if __name__ == "__main__":
    run_inference()
