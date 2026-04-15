import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from pathlib import Path
import json
from transformers import AutoModel
import lightning as L
from sklearn.metrics import silhouette_score

from src.scdino.models.huggingface import ScDINOModel
from src.scdino.utils.conv_mod import conv_mod
from src.scdino.utils.per_channel_wrapper import PerChannelWrapper
from src.scdino.eval.knn import knn_classifier, compute_knn_accuracy


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

    # Load the model
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.datamodule.loader.accelerator == "gpu"
        else "cpu"
    )
    model_id = cfg.local_model_path if cfg.get("local_model_path") else cfg.model.name
    flavor = cfg.get("channel_adaptation_flavor", "mean")
    print(f"Using channel adaptation flavor: {flavor}")
    num_channels = cfg.datamodule.loader.num_channels

    if model_id.startswith("facebook/dinov3"):
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

        def extract_embeddings(model_output):
            return model_output.pooler_output

        if flavor == "DINO4CELL":
            model = PerChannelWrapper(model, extract_embeddings, num_channels)
        else:
            model.embeddings.patch_embeddings = conv_mod(
                model.embeddings.patch_embeddings, num_channels, flavor=flavor,
            )

    elif model_id.startswith("facebook/dinov2"):
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

        def extract_embeddings(model_output):
            return model_output.pooler_output

        if flavor == "DINO4CELL":
            model = PerChannelWrapper(model, extract_embeddings, num_channels)
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
    
    results["silhouette"] = silhouette_score(test_features.cpu().numpy(), 
                                 test_labels.cpu().numpy(), 
                                 sample_size=min(10_000, len(test_features.cpu().numpy())), 
                                 random_state=42)
    print(f"Silhouette score: {results['silhouette']:.4f}")

    # Get output directory
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_dir = Path(out_dir) / "results.json"
    with open(out_dir, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    run_inference()
