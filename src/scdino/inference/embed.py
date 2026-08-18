import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from transformers import AutoModel
import lightning as L

import numpy as np
import pandas as pd
import zarr

from scdino.models.huggingface import ScDINOModel
from scdino.utils.conv_mod import conv_mod
from scdino.utils.per_channel_wrapper import PerChannelWrapper
from scdino.inference.utils import sync_with_training_config


def write_feature_store(
    all_features,
    all_paths,
    out_dir="feature_store",
    chunk_rows=4096,
    seed=42,
):
    all_features = np.asarray(all_features, dtype=np.float32)
    all_paths = np.asarray(all_paths).reshape(-1)

    N, D = all_features.shape
    assert len(all_paths) == N

    # Shuffle once before storage
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    features_shuffled = all_features[perm]
    paths_shuffled = all_paths[perm]

    # Create a 2D Zarr array
    z = zarr.create_array(
        store=f"{out_dir}/features.zarr",
        shape=(N, D),
        chunks=(chunk_rows, D),
        dtype="float32",
        compressors=zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
        ),
    )

    z[:, :] = features_shuffled

    # Parquet for paths
    paths_df = pd.DataFrame(
        {
            "row_idx": np.arange(N),
            "path": paths_shuffled,
            "original_idx": perm,
        }
    )

    paths_df.to_parquet(f"{out_dir}/paths.parquet", index=False)


@hydra.main(config_path="../../../configs", config_name="inference.yaml")
def run_embed(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    sync_with_training_config(cfg)

    # Load the dataset
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("predict")
    train_loader = datamodule.train_dataloader(
        shuffle=False
    )  # no shuffling to access sample paths
    val_loader = datamodule.val_dataloader(
        shuffle=False
    )  # no shuffling to access sample path

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
            model = PerChannelWrapper(
                model, extract_embeddings, num_channels, flavor="mean"
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

        def extract_embeddings(model_output):
            return model_output.pooler_output

        print(f"Using channel adaptation flavor: {flavor}")
        if flavor == "DINO4CELL_CONCAT":
            model = PerChannelWrapper(model, extract_embeddings, num_channels)
        elif flavor == "DINO4CELL_MEAN":
            model = PerChannelWrapper(
                model, extract_embeddings, num_channels, flavor="mean"
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

        def extract_embeddings(model_output):
            return model_output.pooler_output

    if isinstance(model, PerChannelWrapper):

        def embed(images):
            return model(images)
    else:

        def embed(images):
            return extract_embeddings(model(images))

    train_features = []
    test_features = []
    all_paths = []

    i = 0
    for images, _ in tqdm(train_loader):
        with torch.no_grad():
            train_features.append(embed(images.to(device)).detach().cpu())

        samples = train_loader.dataset.samples[i : i + images.shape[0]]
        paths = [s[0] for s in samples]
        all_paths.extend(paths)
        i += images.shape[0]

    i = 0
    for images, _ in tqdm(val_loader):
        with torch.no_grad():
            test_features.append(embed(images.to(device)).detach().cpu())

        samples = val_loader.dataset.samples[i : i + images.shape[0]]
        paths = [s[0] for s in samples]
        all_paths.extend(paths)
        i += images.shape[0]

    train_features = torch.cat(train_features, dim=0)
    test_features = torch.cat(test_features, dim=0)

    all_features = torch.cat([train_features, test_features], dim=0)  # (N, D)

    write_feature_store(
        all_features=all_features,
        all_paths=all_paths,
        out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        chunk_rows=4096,
        seed=42,
    )


if __name__ == "__main__":
    run_embed()


# Downstream usage:


# import zarr
# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import IterableDataset, DataLoader, get_worker_info


# class ZarrFeatureDataset(IterableDataset):
#     def __init__(self, zarr_path, paths_path, shuffle_chunks=True):
#         self.zarr_path = zarr_path
#         self.paths_path = paths_path
#         self.shuffle_chunks = shuffle_chunks

#     def __iter__(self):
#         worker = get_worker_info()

#         x = zarr.open_array(self.zarr_path, mode="r")
#         paths = pd.read_parquet(self.paths_path)["path"].to_numpy()

#         chunk_rows = x.chunks[0]
#         N = x.shape[0]
#         n_chunks = int(np.ceil(N / chunk_rows))

#         chunk_ids = np.arange(n_chunks)

#         if worker is not None:
#             chunk_ids = chunk_ids[worker.id::worker.num_workers]

#         if self.shuffle_chunks:
#             rng = np.random.default_rng()
#             rng.shuffle(chunk_ids)

#         for chunk_id in chunk_ids:
#             start = chunk_id * chunk_rows
#             end = min(start + chunk_rows, N)

#             block = x[start:end, :]
#             block_paths = paths[start:end]

#             for i in range(end - start):
#                 yield torch.from_numpy(block[i]), block_paths[i]

# dataset = ZarrFeatureDataset(
#     zarr_path="feature_store/features.zarr/all_features",
#     paths_path="feature_store/paths.parquet",
# )

# loader = DataLoader(
#     dataset,
#     batch_size=512,
#     num_workers=4,
#     pin_memory=True,
# )

# for features, paths in loader:
#     features = features.cuda(non_blocking=True)
#     # train
