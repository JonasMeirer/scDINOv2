import torch
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from pathlib import Path
import json
from transformers import AutoModel
import lightning as L
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import umap
import pandas as pd

from src.scdino.models.huggingface import ScDINOModel
from src.scdino.utils.conv_mod import conv_mod
from src.scdino.utils.per_channel_wrapper import PerChannelWrapper
from src.scdino.eval.knn import knn_classifier, compute_knn_accuracy
from src.scdino.eval.purity import compute_purity, purity_per_class


@hydra.main(config_path="../../../configs", config_name="inference.yaml")
def run_inference(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    # Load the dataset
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("predict")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader(shuffle=True)

    max_train_batches = cfg.max_train_batches if cfg.max_train_batches is not None else len(train_loader)
    max_val_batches = cfg.max_val_batches if cfg.max_val_batches is not None else len(val_loader)
    n_cells_per_class = cfg.get("n_cells_per_class")
    
    assert n_cells_per_class > 0, "n_cells_per_class must be greater than 0"

    # Load the model
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.datamodule.loader.accelerator == "gpu"
        else "cpu"
    )
    model_id = cfg.local_model_path if cfg.get("local_model_path") else cfg.model.name
    flavor = cfg.get("channel_adaptation_flavor", "mean")
    num_channels = cfg.datamodule.loader.num_channels

    if model_id.startswith("facebook/dinov3"):
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

        def extract_embeddings(model_output):
            return model_output.pooler_output

        print(f"Using channel adaptation flavor: {flavor}")
        if flavor == "DINO4CELL_CONCAT":
            model = PerChannelWrapper(model, extract_embeddings, num_channels)
        elif flavor == "DINO4CELL_MEAN":
            model = PerChannelWrapper(model, extract_embeddings, num_channels, flavor="mean")
        else:
            model.embeddings.patch_embeddings = conv_mod(
                model.embeddings.patch_embeddings, num_channels, flavor=flavor,
            )

    elif model_id.startswith("facebook/dinov2"):
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

        def extract_embeddings(model_output):
            return model_output.pooler_output

        print(f"Using channel adaptation flavor: {flavor}")
        if flavor == "DINO4CELL_CONCAT":
            model = PerChannelWrapper(model, extract_embeddings, num_channels)
        elif flavor == "DINO4CELL_MEAN":
            model = PerChannelWrapper(model, extract_embeddings, num_channels, flavor="mean")
        else:
            model.embeddings.patch_embeddings.projection = conv_mod(
                model.embeddings.patch_embeddings.projection, num_channels, flavor=flavor,
            )
            model.embeddings.patch_embeddings.num_channels = num_channels

    else:
        model = ScDINOModel.from_pretrained(model_id).to(device)
        model.eval()

        def extract_embeddings(model_output):
            return model_output.pooler_output
        
        def get_attn_heatmap(images, target_size):
            heatmap = model.backbone.get_cls_attention_map(images.to(device), head_fusion="none")
            heatmap = F.interpolate(heatmap, size=target_size, mode='bilinear', align_corners=False)
            return heatmap
            

    if isinstance(model, PerChannelWrapper):
        def embed(images):
            return model(images)
    else:
        def embed(images):
            return extract_embeddings(model(images))

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for i, (images, labels) in tqdm(enumerate(train_loader), total=max_train_batches):
        if i >= max_train_batches:
            break
        
        # vizualise images and attn_heatmaps
        if i==0:
            try:
                attn_heatmaps = get_attn_heatmap(images, target_size=images.shape[-2:])
                plt.figure(figsize=(6, 6))
                nrows = 10
                ncols = images.shape[1] + attn_heatmaps.shape[1]
                for j in range(nrows):
                    for c in range(images.shape[1]):
                        idx = j * ncols + c + 1
                        plt.subplot(nrows, ncols, idx)
                        plt.imshow(images[j, c].cpu().numpy(), cmap='gray')
                        plt.axis('off')
                    for c in range(attn_heatmaps.shape[1]):
                        idx = j * ncols + images.shape[1] + c + 1
                        plt.subplot(nrows, ncols, idx)
                        plt.imshow(attn_heatmaps[j, c].cpu().numpy(), cmap='viridis')
                        plt.axis('off')
                plt.tight_layout()
                plt.savefig("attn_heatmap.png")
                plt.show()
            except Exception:
                print(f"Attention heatmap not available for model {model_id}.")
        
        with torch.no_grad():
            train_features.append(embed(images.to(device)).detach().cpu())
            train_labels.append(labels)

    for i, (images, labels) in tqdm(enumerate(val_loader), total=max_val_batches):
        if i >= max_val_batches:
            break
        with torch.no_grad():
            test_features.append(embed(images.to(device)).detach().cpu())
            test_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # unique labels
    unique_labels = torch.unique(train_labels)
    num_classes = len(unique_labels)
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
    print(f"kNN top5 accuracy: {results['top5']:.2f}%")
    
    test_features_np = test_features.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    sample_size = min(10_000, len(test_features_np))

    results["silhouette"] = silhouette_score(test_features_np,
                                 test_labels_np,
                                 sample_size=sample_size,
                                 random_state=42)
    print(f"Silhouette score: {results['silhouette']:.4f}")
    
    
    # for UMAP, sample n_cells_per_class points per class
    test_features_np_sampled = []
    test_labels_np_sampled = []
    for cls in unique_labels.cpu().numpy():
        cls_idx = np.flatnonzero(test_labels.cpu().numpy() == cls)
        take = min(n_cells_per_class, len(cls_idx))
        selected_idx = np.random.choice(cls_idx, size=take, replace=False)
        test_features_np_sampled.append(test_features[selected_idx].cpu().numpy())
        test_labels_np_sampled.append(test_labels[selected_idx].cpu().numpy())
    test_features_np_sampled = np.concatenate(test_features_np_sampled, axis=0)
    test_labels_np_sampled = np.concatenate(test_labels_np_sampled, axis=0)

    umap_cfg = cfg.eval.umap
    reducer = umap.UMAP(
        n_components=umap_cfg.n_components,
        n_neighbors=umap_cfg.n_neighbors,
        min_dist=umap_cfg.min_dist,
        metric=umap_cfg.metric,
    )
    umap_features = reducer.fit_transform(test_features_np_sampled)
    results["silhouette_umap"] = silhouette_score(umap_features,
                                 test_labels_np_sampled,
                                 random_state=42)
    print(f"Silhouette score (UMAP): {results['silhouette_umap']:.4f}")

    y = test_labels_np_sampled
    y_class = pd.Series(y).map({val: key for key, val in datamodule.val_dataset.class_to_idx.items()}).values

    ks = (1, 10, 100, 1000)
    same_label, purities = compute_purity(test_features_np_sampled, y, ks=ks)
    same_label_umap, purities_umap = compute_purity(umap_features, y, ks=ks)
    for k in ks:
        results[f"purity@{k}"] = purities[k]
        results[f"purity_umap@{k}"] = purities_umap[k]
        print(f"Purity@{k}: {purities[k]:.3f}  |  Purity_umap@{k}: {purities_umap[k]:.3f}")

    print("Purity@100 per class:")
    per_class = purity_per_class(same_label, y_class, k=100)
    per_class_umap = purity_per_class(same_label_umap, y_class, k=100)
    results["val_purity@100_classes"] = {f"class_{cls}": format(v, ".4f") for cls, v in per_class.items()}
    for cls, v in per_class.items():
        print("\t", cls, format(v, ".4f"))
    results["val_purity_umap@100_classes"] = {f"class_{cls}": format(v, ".4f") for cls, v in per_class_umap.items()}
    for cls, v in per_class_umap.items():
        print("\t", cls, format(v, ".4f"))

    hydra_out = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    n_per_class = cfg.eval.viz.cells_per_class
    class_names = getattr(datamodule.val_dataset, "classes", None)

    rng = np.random.default_rng(cfg.seed)
    viz_idx = []
    for cls in np.unique(test_labels_np_sampled):
        cls_idx = np.flatnonzero(test_labels_np_sampled == cls)
        take = min(n_per_class, len(cls_idx))
        viz_idx.append(rng.choice(cls_idx, size=take, replace=False))
    viz_idx = np.concatenate(viz_idx)
    viz_pts = umap_features[viz_idx, :2]
    viz_lab = test_labels_np_sampled[viz_idx]

    unique = np.unique(viz_lab)
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, cls in enumerate(unique):
        m = viz_lab == cls
        label = class_names[int(cls)] if class_names is not None else str(int(cls))
        ax.scatter(viz_pts[m, 0], viz_pts[m, 1], s=4, alpha=0.6,
                   color=cmap(i), label=label)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP of test embeddings (≤{n_per_class}/class)")
    ax.legend(markerscale=3, loc="best", fontsize=9)
    plt.tight_layout()
    viz_path = hydra_out / "umap.png"
    plt.savefig(viz_path, dpi=150)
    plt.close(fig)
    print(f"Saved UMAP visualization to {viz_path}")

    # Get output directory
    out_dir = hydra_out / "results.json"
    results = {
        "seed": cfg.seed,
        "metrics": {"val_knn_top1": format(results['top1'], ".4f"),
                    "val_knn_top5": format(results['top5'], ".4f"),
                    "val_silhouette": format(results['silhouette'], ".4f"),
                    "val_silhouette_umap": format(results['silhouette_umap'], ".4f"),
                    "val_purity@1": format(results['purity@1'], ".4f"),
                    "val_purity@10": format(results['purity@10'], ".4f"),
                    "val_purity@100": format(results['purity@100'], ".4f"),
                    "val_purity@1000": format(results['purity@1000'], ".4f"),
                    "val_purity_umap@1": format(results['purity_umap@1'], ".4f"),
                    "val_purity_umap@10": format(results['purity_umap@10'], ".4f"),
                    "val_purity_umap@100": format(results['purity_umap@100'], ".4f"),
                    "val_purity_umap@1000": format(results['purity_umap@1000'], ".4f"),
                    "val_purity@100_classes": results['val_purity@100_classes'],
                    "val_purity_umap@100_classes": results['val_purity_umap@100_classes'],
                    },
    }
    with open(out_dir, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    run_inference()
