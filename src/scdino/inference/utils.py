"""Helpers for inference: model construction, embedding extraction, viz, clustering."""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import AutoModel

from src.scdino.models.huggingface import ScDINOModel
from src.scdino.utils.conv_mod import conv_mod
from src.scdino.utils.per_channel_wrapper import PerChannelWrapper


def build_model(model_id, flavor, num_channels, device):
    """Construct the embedding model and (optionally) an attention-heatmap fn.

    Returns:
        embed: callable mapping a batch of images on `device` to embeddings.
        attn_fn: callable producing a CLS-attention heatmap, or None if the
            model does not expose one (only ScDINOModel does).
    """
    attn_fn = None

    if model_id.startswith("facebook/dinov3"):
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        print(f"Using channel adaptation flavor: {flavor}")
        if flavor == "DINO4CELL_CONCAT":
            model = PerChannelWrapper(model, _pooler_output, num_channels)
        elif flavor == "DINO4CELL_MEAN":
            model = PerChannelWrapper(model, _pooler_output, num_channels, flavor="mean")
        else:
            model.embeddings.patch_embeddings = conv_mod(
                model.embeddings.patch_embeddings, num_channels, flavor=flavor,
            )

    elif model_id.startswith("facebook/dinov2"):
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        print(f"Using channel adaptation flavor: {flavor}")
        if flavor == "DINO4CELL_CONCAT":
            model = PerChannelWrapper(model, _pooler_output, num_channels)
        elif flavor == "DINO4CELL_MEAN":
            model = PerChannelWrapper(model, _pooler_output, num_channels, flavor="mean")
        else:
            model.embeddings.patch_embeddings.projection = conv_mod(
                model.embeddings.patch_embeddings.projection, num_channels, flavor=flavor,
            )
            model.embeddings.patch_embeddings.num_channels = num_channels

    else:
        model = ScDINOModel.from_pretrained(model_id).to(device)
        model.eval()
        backbone = model.backbone

        def attn_fn(images, target_size):
            heatmap = backbone.get_cls_attention_map(images.to(device), head_fusion="none")
            return F.interpolate(heatmap, size=target_size, mode="bilinear", align_corners=False)

    if isinstance(model, PerChannelWrapper):
        def embed(images):
            return model(images)
    else:
        def embed(images):
            return _pooler_output(model(images))

    return embed, attn_fn


def _pooler_output(model_output):
    return model_output.pooler_output


def save_attention_heatmap_grid(images, attn_heatmaps, path, nrows=10):
    """Save a row-per-sample grid of image channels alongside attention heads."""
    ncols = images.shape[1] + attn_heatmaps.shape[1]
    plt.figure(figsize=(6, 6))
    for j in range(nrows):
        for c in range(images.shape[1]):
            plt.subplot(nrows, ncols, j * ncols + c + 1)
            plt.imshow(images[j, c].cpu().numpy(), cmap="gray")
            plt.axis("off")
        for c in range(attn_heatmaps.shape[1]):
            plt.subplot(nrows, ncols, j * ncols + images.shape[1] + c + 1)
            plt.imshow(attn_heatmaps[j, c].cpu().numpy(), cmap="viridis")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def extract_features(loader, embed, device, max_batches, attn_fn=None, attn_path=None):
    """Run `embed` over `loader`, timing only the forward pass.

    Returns (features, labels, n_samples, seconds). If `attn_fn` and `attn_path`
    are provided, a heatmap visualization is saved from the first batch.
    """
    features, labels_list = [], []
    n_samples, seconds = 0, 0.0
    for i, (images, labels) in tqdm(enumerate(loader), total=max_batches):
        if i >= max_batches:
            break

        if i == 0 and attn_fn is not None and attn_path is not None:
            try:
                attn = attn_fn(images, target_size=images.shape[-2:])
                save_attention_heatmap_grid(images, attn, attn_path)
            except Exception as e:
                print(f"Attention heatmap unavailable: {e}")

        with torch.no_grad():
            images_dev = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            feats = embed(images_dev)
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds += time.perf_counter() - t0
            n_samples += images_dev.shape[0]
            features.append(feats.detach().cpu())
            labels_list.append(labels)

    return (
        torch.cat(features, dim=0),
        torch.cat(labels_list, dim=0),
        n_samples,
        seconds,
    )


def sample_per_class(labels_np, n_per_class, rng=None):
    """Return concatenated indices, up to `n_per_class` per unique class."""
    choice = rng.choice if rng is not None else np.random.choice
    out = []
    for cls in np.unique(labels_np):
        cls_idx = np.flatnonzero(labels_np == cls)
        take = min(n_per_class, len(cls_idx))
        out.append(choice(cls_idx, size=take, replace=False))
    return np.concatenate(out)


def plot_umap_by_class(viz_pts, viz_lab, n_per_class, class_names, path):
    unique = np.unique(viz_lab)
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, cls in enumerate(unique):
        m = viz_lab == cls
        label = class_names[int(cls)] if class_names is not None else str(int(cls))
        ax.scatter(viz_pts[m, 0], viz_pts[m, 1], s=4, alpha=0.6, color=cmap(i), label=label)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP of test embeddings (≤{n_per_class}/class)")
    ax.legend(markerscale=3, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_umap_correctness(viz_pts, viz_correct, n_per_class, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(viz_pts[viz_correct, 0], viz_pts[viz_correct, 1],
               s=4, alpha=0.6, color="lightgray", label="correct")
    ax.scatter(viz_pts[~viz_correct, 0], viz_pts[~viz_correct, 1],
               s=4, alpha=0.6, color="red", label="incorrect")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP of test embeddings — kNN correctness (≤{n_per_class}/class)")
    ax.legend(markerscale=3, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run_hdbscan(features, hdbscan_cfg, plot_path=None):
    """Cluster `features` with HDBSCAN and report metrics.

    Returns {"n_clusters": int, "silhouette": float (or NaN)}. If `plot_path` is
    provided and at least 2 clusters are found, saves a hierarchically-ordered
    cluster-centroid distance heatmap.
    """
    import hdbscan
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist, squareform

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_cfg.min_cluster_size,
        min_samples=hdbscan_cfg.min_samples,
        metric=hdbscan_cfg.metric,
    )
    labels = clusterer.fit_predict(features)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points "
          f"({n_noise / len(labels) * 100:.1f}%)")

    mask = labels != -1
    if n_clusters >= 2 and mask.sum() >= 2:
        sil = float(silhouette_score(features[mask], labels[mask], random_state=42))
        print(f"HDBSCAN silhouette score: {sil:.4f}")
    else:
        sil = float("nan")
        print("HDBSCAN silhouette score: undefined (fewer than 2 clusters)")

    if plot_path is not None and n_clusters >= 2:
        cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)
        centroids = np.stack([features[labels == cid].mean(axis=0) for cid in cluster_ids])
        dists = pdist(centroids, metric=hdbscan_cfg.metric)
        Z = linkage(dists, method="average")
        order = leaves_list(Z)
        D = squareform(dists)[order][:, order]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(D, cmap="viridis")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels([cluster_ids[i] for i in order], rotation=90, fontsize=7)
        ax.set_yticklabels([cluster_ids[i] for i in order], fontsize=7)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Cluster ID")
        ax.set_title("HDBSCAN cluster centroid distances (hierarchical order)")
        fig.colorbar(im, ax=ax, label=f"{hdbscan_cfg.metric} distance")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved HDBSCAN cluster distance matrix to {plot_path}")

    return {"n_clusters": n_clusters, "silhouette": sil}
