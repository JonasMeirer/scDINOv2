"""Helpers for inference: model construction, embedding extraction, viz, clustering."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.metrics import confusion_matrix, silhouette_score
from tqdm import tqdm
from transformers import AutoModel

from src.scdino.models.huggingface import ScDINOModel
from src.scdino.utils.conv_mod import conv_mod
from src.scdino.utils.per_channel_wrapper import PerChannelWrapper


def sync_with_training_config(cfg: DictConfig) -> None:
    """For a local scDINO checkpoint, override model/preprocessing on `cfg` to match training.

    No-op when no `local_model_path` is set (pretrained `facebook/...` path) or when
    `cfg.model.name` does not start with "dino". Otherwise reads the training run's
    `.hydra/config.yaml` (sibling of the checkpoint dir) and overwrites fields that
    affect model architecture and image preprocessing. Hardware/runtime fields, eval
    settings, paths, seed, and mode are left untouched.
    """
    local_path = cfg.get("local_model_path")
    if not local_path:
        return
    if not cfg.model.name.startswith("dino"):
        return

    train_cfg_path = Path(local_path).parent / ".hydra" / "config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(
            f"Expected training config at {train_cfg_path} for local_model_path={local_path}"
        )
    train_cfg = OmegaConf.load(train_cfg_path)

    with open_dict(cfg):
        cfg.model = train_cfg.model
        cfg.datamodule.transforms = train_cfg.datamodule.transforms
        for key in ("num_channels", "norm_type", "norm_dict", "max_vals_clip"):
            cfg.datamodule.loader[key] = train_cfg.datamodule.loader[key]

    print(f"Synced model + preprocessing config from {train_cfg_path}")

    out_cfg_path = Path(HydraConfig.get().runtime.output_dir) / ".hydra" / "config.yaml"
    OmegaConf.save(cfg, out_cfg_path)
    print(f"Saved synced config to {out_cfg_path}")


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
            model = PerChannelWrapper(
                model, _pooler_output, num_channels, flavor="mean"
            )
        else:
            model.embeddings.patch_embeddings = conv_mod(
                model.embeddings.patch_embeddings,
                num_channels,
                flavor=flavor,
            )

    elif model_id.startswith("facebook/dinov2"):
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        print(f"Using channel adaptation flavor: {flavor}")
        if flavor == "DINO4CELL_CONCAT":
            model = PerChannelWrapper(model, _pooler_output, num_channels)
        elif flavor == "DINO4CELL_MEAN":
            model = PerChannelWrapper(
                model, _pooler_output, num_channels, flavor="mean"
            )
        else:
            model.embeddings.patch_embeddings.projection = conv_mod(
                model.embeddings.patch_embeddings.projection,
                num_channels,
                flavor=flavor,
            )
            model.embeddings.patch_embeddings.num_channels = num_channels

    else:
        model = ScDINOModel.from_pretrained(model_id).to(device)
        model.eval()
        backbone = model.backbone

        def attn_fn(images, target_size):
            heatmap = backbone.get_cls_attention_map(
                images.to(device), head_fusion="none"
            )
            return F.interpolate(
                heatmap, size=target_size, mode="bilinear", align_corners=False
            )

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
        ax.scatter(
            viz_pts[m, 0], viz_pts[m, 1], s=4, alpha=0.6, color=cmap(i), label=label
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP of test embeddings (≤{n_per_class}/class)")
    ax.legend(markerscale=3, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_umap_correctness(viz_pts, viz_correct, n_per_class, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        viz_pts[viz_correct, 0],
        viz_pts[viz_correct, 1],
        s=4,
        alpha=0.6,
        color="lightgray",
        label="correct",
    )
    ax.scatter(
        viz_pts[~viz_correct, 0],
        viz_pts[~viz_correct, 1],
        s=4,
        alpha=0.6,
        color="red",
        label="incorrect",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP of test embeddings — kNN correctness (≤{n_per_class}/class)")
    ax.legend(markerscale=3, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, path, normalize=True):
    """Save a confusion matrix heatmap. If `normalize`, rows sum to 1 (recall per class)."""
    labels = np.arange(len(class_names)) if class_names is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_disp = np.nan_to_num(cm_disp)
        vmin, vmax, fmt = 0.0, 1.0, ".2f"
        cbar_label = "Fraction (row-normalized)"
        title = "Confusion matrix (row-normalized)"
    else:
        cm_disp = cm
        vmin, vmax, fmt = 0, cm.max() if cm.size else 1, "d"
        cbar_label = "Count"
        title = "Confusion matrix"

    n = cm_disp.shape[0]
    tick_labels = (
        [class_names[i] for i in range(n)]
        if class_names is not None
        else [str(i) for i in range(n)]
    )
    side = max(6, 0.5 * n + 2)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm_disp, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    if n <= 30:
        thresh = (vmin + vmax) / 2.0 if normalize else cm_disp.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    format(cm_disp[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm_disp[i, j] > thresh else "black",
                )

    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
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
        cluster_selection_method=hdbscan_cfg.cluster_selection_method,
    )
    if hdbscan_cfg.l2_normalize:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = clusterer.fit_predict(features)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int((labels == -1).sum())
    print(
        f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points "
        f"({n_noise / len(labels) * 100:.1f}%)"
    )

    mask = labels != -1
    if n_clusters >= 2 and mask.sum() >= 2:
        sil = float(silhouette_score(features[mask], labels[mask], random_state=42))
        print(f"HDBSCAN silhouette score: {sil:.4f}")
    else:
        sil = float("nan")
        print("HDBSCAN silhouette score: undefined (fewer than 2 clusters)")

    if plot_path is not None and n_clusters >= 2:
        cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)
        centroids = np.stack(
            [features[labels == cid].mean(axis=0) for cid in cluster_ids]
        )
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
