import lightning as L

import torch
import torchvision.transforms as T
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

import tifffile
import numpy as np
from omegaconf import ListConfig

from src.scdino.data.transforms.trafo import Trafo


class CHRONOTYPEDataModule(L.LightningDataModule):
    def __init__(self, dataset, model, mode, paths, loader, transforms):
        super().__init__()

        self.name = dataset
        # PATHS
        self.data_dir_train = paths.get("train_dir", "/mnt/SSD/Chronotype/train_all")
        self.data_dir_val = paths.get("val_dir", "/mnt/SSD/Chronotype/val")
        self.test_dir = paths.get("test_dir", None)
        self.predict_dir = paths.get("predict_dir", None)

        # Dtaloader
        self.batch_size = loader.get("batch_size", 32)
        self.shuffle = loader.get("shuffle", True)
        self.drop_last = loader.get("drop_last", True)
        self.num_workers = loader.get("num_workers", 24)
        self.prefetch_factor = loader.get("prefetch_factor", 2)
        self.pin_memory = loader.get("pin_memory", True)

        # TRANSFORMS
        self.norm_type = loader.get("norm_type", "robust")
        self.max_vals_clip = loader.get("max_vals_clip", None)
        mean_raw = loader.norm_dict.get("raw").get("mean")
        std_raw = loader.norm_dict.get("raw").get("std")
        mean_norm = loader.norm_dict.get(self.norm_type).get("mean")
        std_norm = loader.norm_dict.get(self.norm_type).get("std")
        transforms.normalize = {"mean": mean_norm, "std": std_norm} # needed for Trafo
        self.resize = transforms.get("resize", None)
        self.train_transform = Trafo(model, mode, transforms)
        self.norm_only_transform = T.Compose(
            [
                T.Lambda(lambda x: x[0] if isinstance(x, list) else x),
                T.Normalize(mean=mean_raw, std=std_raw),
            ]
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = DatasetFolder(
                root=self.data_dir_train,
                loader=self.load_tiff,
                extensions=(".tiff",),
                transform=self.train_transform,
                target_transform=None,
            )

            self.val_dataset = DatasetFolder(
                root=self.data_dir_val,
                loader=self.load_tiff,
                extensions=(".tiff",),
                transform=self.norm_only_transform,
                target_transform=None,
            )

        elif stage == "test":
            if self.test_dir is not None:
                self.test_dataset = DatasetFolder(
                    root=self.test_dir,
                    loader=self.load_tiff,
                    extensions=(".tiff",),
                    transform=self.norm_only_transform,
                    target_transform=None,
                )
            else:
                print("Using val_dataset for test")
                self.test_dataset = self.val_dataset

        elif stage == "predict":
            self.train_dataset = DatasetFolder(
                root=self.data_dir_train,
                loader=self.load_tiff,
                extensions=(".tiff",),
                transform=self.norm_only_transform,
                target_transform=None,
            )

            self.val_dataset = DatasetFolder(
                root=self.data_dir_val,
                loader=self.load_tiff,
                extensions=(".tiff",),
                transform=self.norm_only_transform,
                target_transform=None,
            )
            if isinstance(self.predict_dir, str):
                self.predict_datasets = [
                    DatasetFolder(
                        root=self.predict_dir,
                        loader=self.load_tiff,
                        extensions=(".tiff",),
                        transform=self.norm_only_transform,
                        target_transform=None,
                    )
                ]
            elif isinstance(self.predict_dir, ListConfig):
                self.predict_datasets = []
                for dir in self.predict_dir:
                    self.predict_datasets.append(
                        DatasetFolder(
                            root=dir,
                            loader=self.load_tiff,
                            extensions=(".tiff",),
                            transform=self.norm_only_transform,
                            target_transform=None,
                        )
                    )
            else:
                raise ValueError("predict_dir must be a string or a list of strings")

        elif stage == "calibrate":
            self.train_dataset = DatasetFolder(
                root=self.data_dir_train,
                loader=self.load_tiff,
                extensions=(".tiff",),
                transform=None,
                target_transform=None,
            )

            self.val_dataset = DatasetFolder(
                root=self.data_dir_val,
                loader=self.load_tiff,
                extensions=(".tiff",),
                transform=None,
                target_transform=None,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        loaders = []
        for dataset in self.predict_datasets:
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    pin_memory=self.pin_memory,
                )
            )

        return loaders

    def clip(self, x):
        x = np.clip(x, max=self.max_vals_clip)
        return x

    def load_tiff(self, path):
        img = tifffile.imread(path)  # (50, 50, 5)

        if self.norm_type == "robust":
            img = normalize_numpy_robust(img)
        elif self.norm_type == "robust_2":
            img = normalize_numpy_robust_2(img)
        elif self.norm_type == "clip_max":
            img = self.clip(img)
        else:
            raise ValueError(f"Invalid norm type: {self.norm_type}")

        img = img.T  # (5, 50, 50)
        img = torch.from_numpy(img).float()
        if self.resize is not None:
            img = T.Resize(size=self.resize)(img)
        return img


# HELPERS


def normalize_numpy_robust(x):
    # removes the median and scales the data according to the quantile range
    # the quantile range is defined as the 75th quantile - 25th quantile
    # x.shape = (50, 50, 50)
    x_median = np.median(x, axis=(0, 1))
    x_q25 = np.quantile(x, 0.025, axis=(0, 1))
    x_q75 = np.quantile(x, 0.975, axis=(0, 1))
    x_qrange = x_q75 - x_q25
    # if set any zeros in q_range to 1
    x_qrange = np.where(x_qrange < 1, 1, x_qrange)
    x = (x - x_median) / x_qrange
    return x


def normalize_numpy_robust_2(x, min_range=1):
    """
    Robust per-channel normalization for microscopy data.

    Parameters
    ----------
    x : np.ndarray
        Input array with shape (H, W, C)
    min_range : float
        Minimum dynamic range required to treat a channel as containing signal.

    Returns
    -------
    np.ndarray
        Normalized array with shape (H, W, C)
    """

    x = np.log1p(x.astype(np.float32))

    # robust percentiles per channel
    lo = np.quantile(x, 0.01, axis=(0, 1))
    hi = np.quantile(x, 0.99, axis=(0, 1))

    # clip outliers
    # x = np.clip(x, lo, hi)

    # dynamic range per channel
    den = hi - lo

    # avoid divide-by-zero while preventing noise amplification
    safe_den = np.maximum(den, min_range)

    # normalize
    x = (x - lo) / safe_den

    # channels with insufficient range → treat as absent signal
    low_signal = den < min_range
    if np.any(low_signal):
        x[..., low_signal] = 0

    return x
