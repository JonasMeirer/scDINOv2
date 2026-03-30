import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from pathlib import Path
import json
from transformers import AutoModel

from src.scdino.utils.conv_mod import conv_mod
from src.scdino.eval.knn import knn_classifier, compute_knn_accuracy


@hydra.main(config_path="../../../configs", config_name="inference.yaml")
def run_inference(cfg: DictConfig):

    # Load the dataset
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("predict")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    max_batches = cfg.max_batches if cfg.max_batches is not None else len(train_loader)

    # Load the model
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.datamodule.loader.accelerator == "gpu"
        else "cpu"
    )
    model = AutoModel.from_pretrained(cfg.model.name).to(device)
    model.eval()

    if cfg.model.name.startswith("facebook/dinov3"):
        # modify the patch embeddings to the correct number of input channels
        model.embeddings.patch_embeddings = conv_mod(
            model.embeddings.patch_embeddings, cfg.datamodule.loader.num_channels
        )

        def extract_embeddings(model_output):
            return model_output.pooler_output

    elif cfg.model.name.startswith("facebook/dinov2"):
        model.embeddings.patch_embeddings.projection = conv_mod(
            model.embeddings.patch_embeddings.projection,
            cfg.datamodule.loader.num_channels,
        )
        model.embeddings.patch_embeddings.num_channels = 5

        def extract_embeddings(model_output):
            return model_output.pooler_output

    else:
        raise ValueError(f"Model {cfg.model.name} not supported")

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for i, (images, labels) in tqdm(enumerate(train_loader), total=max_batches):
        if i >= cfg.max_batches:
            break
        with torch.no_grad():
            train_features.append(
                extract_embeddings(model(images.to(device))).detach().cpu()
            )
            train_labels.append(labels)

    for images, labels in tqdm(val_loader):
        with torch.no_grad():
            test_features.append(
                extract_embeddings(model(images.to(device))).detach().cpu()
            )
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

    # Get output directory
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_dir = Path(out_dir) / "results.json"
    with open(out_dir, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    run_inference()
