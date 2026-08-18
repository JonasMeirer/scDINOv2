"""Tests for dataset normalisation and the datamodule's path handling."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scdino.data.chronotype import (
    CHRONOTYPEDataModule,
    normalize_numpy_robust,
    normalize_numpy_robust_2,
)


def make_loader_cfg(**overrides):
    cfg = {
        "batch_size": 4,
        "num_workers": 0,
        "prefetch_factor": None,
        "pin_memory": False,
        "num_channels": 5,
        "norm_type": "clip_max",
        "norm_dict": {
            "clip_max": {"mean": [1.0] * 5, "std": [2.0] * 5},
            "robust": {"mean": [0.0] * 5, "std": [1.0] * 5},
            "identity": {"mean": [0.0] * 5, "std": [1.0] * 5},
            "not-a-norm": {"mean": [0.0] * 5, "std": [1.0] * 5},
        },
        "max_vals_clip": [100.0] * 5,
        "max_train_samples": None,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def make_datamodule(tmp_path, **loader_overrides):
    train, val = tmp_path / "train", tmp_path / "val"
    train.mkdir(exist_ok=True)
    val.mkdir(exist_ok=True)
    return CHRONOTYPEDataModule(
        dataset="test",
        model=OmegaConf.create({"name": "dinov2"}),
        mode="inference",
        paths=OmegaConf.create({"train_dir": str(train), "val_dir": str(val)}),
        loader=make_loader_cfg(**loader_overrides),
        transforms=OmegaConf.create({"normalize": None, "resize": None}),
    )


class TestNormalizeRobust:
    def test_centres_on_the_median(self):
        x = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (8, 8, 1))
        out = normalize_numpy_robust(x)
        assert np.allclose(out, 0.0)

    def test_preserves_shape(self):
        out = normalize_numpy_robust(np.random.rand(8, 8, 5) * 100)
        assert out.shape == (8, 8, 5)

    def test_constant_channel_does_not_divide_by_zero(self):
        out = normalize_numpy_robust(np.ones((8, 8, 5)))
        assert np.isfinite(out).all()


class TestNormalizeRobust2:
    def test_maps_signal_into_roughly_the_unit_interval(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10_000, size=(32, 32, 5))
        out = normalize_numpy_robust_2(x)
        assert out.shape == (32, 32, 5)
        assert np.isfinite(out).all()
        # 1st..99th percentile is mapped to [0, 1]; tails may fall outside.
        assert -0.5 < np.median(out) < 1.5

    def test_low_dynamic_range_channel_is_zeroed(self):
        """A channel with no usable signal is treated as absent, not amplified."""
        x = np.random.rand(16, 16, 5) * 10_000
        x[..., 2] = 5.0  # flat channel
        out = normalize_numpy_robust_2(x)
        assert np.allclose(out[..., 2], 0.0)
        assert not np.allclose(out[..., 0], 0.0)

    def test_is_monotone_within_a_channel(self):
        x = np.zeros((4, 4, 1))
        x[..., 0] = np.arange(16).reshape(4, 4) * 100
        out = normalize_numpy_robust_2(x)
        flat = out[..., 0].ravel()
        assert np.all(np.diff(flat) >= -1e-9)

    def test_returns_float32(self):
        out = normalize_numpy_robust_2(np.random.randint(0, 5000, size=(8, 8, 5)))
        assert out.dtype == np.float32


class TestDataModulePaths:
    def test_requires_a_train_directory(self, tmp_path):
        with pytest.raises(ValueError, match="train_dir"):
            CHRONOTYPEDataModule(
                dataset="test",
                model=OmegaConf.create({"name": "dinov2"}),
                mode="inference",
                paths=OmegaConf.create({"val_dir": str(tmp_path)}),
                loader=make_loader_cfg(),
                transforms=OmegaConf.create({"normalize": None}),
            )

    def test_requires_a_val_directory(self, tmp_path):
        with pytest.raises(ValueError, match="val_dir"):
            CHRONOTYPEDataModule(
                dataset="test",
                model=OmegaConf.create({"name": "dinov2"}),
                mode="inference",
                paths=OmegaConf.create({"train_dir": str(tmp_path)}),
                loader=make_loader_cfg(),
                transforms=OmegaConf.create({"normalize": None}),
            )

    def test_error_message_mentions_how_to_fix_it(self, tmp_path):
        """A missing DATA_DIR is the most common first-run failure."""
        with pytest.raises(ValueError, match="DATA_DIR"):
            CHRONOTYPEDataModule(
                dataset="test",
                model=OmegaConf.create({"name": "dinov2"}),
                mode="inference",
                paths=OmegaConf.create({}),
                loader=make_loader_cfg(),
                transforms=OmegaConf.create({"normalize": None}),
            )

    def test_no_hardcoded_machine_specific_fallback(self, tmp_path):
        """Paths must come from config, never from a developer's local disk."""
        dm = make_datamodule(tmp_path)
        assert "/mnt/SSD" not in str(dm.data_dir_train)
        assert "/mnt/SSD" not in str(dm.data_dir_val)


class TestLoadTiff:
    def test_clip_max_bounds_the_output(self, tmp_path):
        import tifffile

        dm = make_datamodule(tmp_path, norm_type="clip_max", max_vals_clip=[10.0] * 5)
        path = tmp_path / "crop.tiff"
        tifffile.imwrite(path, np.full((8, 8, 5), 1000.0, dtype=np.float32))
        out = dm.load_tiff(str(path))
        assert isinstance(out, torch.Tensor)
        assert out.max().item() <= 10.0

    def test_returns_channels_first(self, tmp_path):
        import tifffile

        dm = make_datamodule(tmp_path, norm_type="identity")
        path = tmp_path / "crop.tiff"
        tifffile.imwrite(path, np.zeros((8, 8, 5), dtype=np.float32))
        assert dm.load_tiff(str(path)).shape[0] == 5

    def test_preserves_spatial_orientation(self, tmp_path):
        """(H, W, C) -> (C, H, W) must not transpose the image.

        Regression test: `ndarray.T` reverses *every* axis and yields
        (C, W, H), silently swapping height and width. A non-square,
        non-symmetric crop is the only way to see it.
        """
        import tifffile

        dm = make_datamodule(tmp_path, norm_type="identity")
        hwc = np.arange(4 * 6 * 5, dtype=np.float32).reshape(4, 6, 5)  # H=4, W=6, C=5
        path = tmp_path / "crop.tiff"
        tifffile.imwrite(path, hwc)

        out = dm.load_tiff(str(path)).numpy()
        assert out.shape == (5, 4, 6), "expected (C, H, W)"
        for channel in range(5):
            np.testing.assert_allclose(out[channel], hwc[:, :, channel])

    def test_rejects_an_unknown_norm_type(self, tmp_path):
        import tifffile

        dm = make_datamodule(tmp_path, norm_type="not-a-norm")
        path = tmp_path / "crop.tiff"
        tifffile.imwrite(path, np.zeros((8, 8, 5), dtype=np.float32))
        with pytest.raises(ValueError, match="Invalid norm type"):
            dm.load_tiff(str(path))
