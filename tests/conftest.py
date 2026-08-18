"""Shared fixtures.

Everything here is CPU-only and small enough to run in a few seconds so the
suite stays usable as a pre-commit / CI gate.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT_CONFIGS = "configs"


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed every test. Tests that need a specific seed can reseed themselves."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def tiny_vit_backbone_config():
    """A ViT backbone config small enough to instantiate in-process."""
    return OmegaConf.create(
        {
            "type": "vit",
            "vit": {
                "in_chans": 5,
                "img_size": 16,
                "patch_size": 4,
                "embed_dim": 16,
                "depth": 2,
                "num_heads": 2,
                "drop_path_rate": 0.1,
                "mlp_ratio": 4,
                "reg_tokens": 2,
            },
        }
    )


@pytest.fixture
def tiny_head_config():
    return OmegaConf.create(
        {
            "embed_dim": 16,
            "hidden_dim": 16,
            "bottleneck_dim": 8,
            "output_dim": 32,
            "batch_norm": False,
        }
    )


@pytest.fixture
def labelled_features():
    """Well-separated 3-class features: kNN should be perfect on these."""
    torch.manual_seed(0)
    centres = torch.eye(3) * 10.0
    feats, labels = [], []
    for cls in range(3):
        feats.append(centres[cls] + 0.01 * torch.randn(40, 3))
        labels.append(torch.full((40,), cls, dtype=torch.long))
    return torch.cat(feats), torch.cat(labels)
