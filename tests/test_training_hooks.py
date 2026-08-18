"""Tests for scDINO's Lightning training hooks.

The last-layer freeze is the interesting one. DINO randomly initialises the
projection head, so its first-epoch gradients are large and noisy; the
reference recipe drops them for the first epoch. An earlier implementation
looked for a `"last_layer"` key in `optimizer.param_groups`, which never
matches because param groups only carry hyperparameters, so nothing was ever
frozen.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.scdino.models.backbones.dinov2 import DINOv2ProjectionHead
from src.scdino.models.lightning.hooks import freeze_last_layer_gradients


def make_head(**kwargs):
    return DINOv2ProjectionHead(
        input_dim=8, hidden_dim=8, bottleneck_dim=4, output_dim=16, **kwargs
    )


def populate_gradients(head, x=None):
    """Run a backward pass so every parameter has a gradient."""
    x = torch.randn(4, 8) if x is None else x
    head(x).sum().backward()


class TestFreezeLastLayerGradients:
    def test_drops_last_layer_gradients_during_warmup(self):
        head = make_head()
        populate_gradients(head)
        assert any(p.grad is not None for p in head.last_layer.parameters())

        freeze_last_layer_gradients(head, current_epoch=0, freeze_epochs=1)
        assert all(p.grad is None for p in head.last_layer.parameters())

    def test_leaves_the_rest_of_the_head_trainable(self):
        """Only the output layer is frozen; the MLP must keep learning."""
        head = make_head()
        populate_gradients(head)
        freeze_last_layer_gradients(head, current_epoch=0, freeze_epochs=1)
        assert any(p.grad is not None for p in head.layers.parameters())

    def test_is_a_no_op_after_the_warmup_epochs(self):
        head = make_head()
        populate_gradients(head)
        freeze_last_layer_gradients(head, current_epoch=1, freeze_epochs=1)
        assert any(p.grad is not None for p in head.last_layer.parameters())

    @pytest.mark.parametrize("epoch,frozen", [(0, True), (1, True), (2, False)])
    def test_respects_a_longer_freeze_window(self, epoch, frozen):
        head = make_head()
        populate_gradients(head)
        freeze_last_layer_gradients(head, current_epoch=epoch, freeze_epochs=2)
        is_frozen = all(p.grad is None for p in head.last_layer.parameters())
        assert is_frozen is frozen

    def test_zero_freeze_epochs_disables_freezing(self):
        head = make_head()
        populate_gradients(head)
        freeze_last_layer_gradients(head, current_epoch=0, freeze_epochs=0)
        assert any(p.grad is not None for p in head.last_layer.parameters())

    def test_handles_a_container_with_tied_dino_and_ibot_heads(self):
        """With ibot_separate_head=False both attributes are the same module."""
        from src.scdino.models.backbones.dinov2 import DINOv2Head

        shared = make_head()
        container = DINOv2Head(dino_head=shared, ibot_head=shared)
        populate_gradients(shared)

        freeze_last_layer_gradients(container, current_epoch=0, freeze_epochs=1)
        assert all(p.grad is None for p in shared.last_layer.parameters())

    def test_handles_a_container_with_separate_dino_and_ibot_heads(self):
        from src.scdino.models.backbones.dinov2 import DINOv2Head

        dino_head, ibot_head = make_head(), make_head()
        container = DINOv2Head(dino_head=dino_head, ibot_head=ibot_head)
        populate_gradients(dino_head)
        populate_gradients(ibot_head)

        freeze_last_layer_gradients(container, current_epoch=0, freeze_epochs=1)
        assert all(p.grad is None for p in dino_head.last_layer.parameters())
        assert all(p.grad is None for p in ibot_head.last_layer.parameters())

    def test_rejects_a_module_with_no_recognisable_head(self):
        with pytest.raises(AttributeError, match="last_layer"):
            freeze_last_layer_gradients(nn.Linear(4, 4), current_epoch=0)


class TestDINOv2WarmupIntegration:
    """The hook must actually be wired into the LightningModule."""

    @pytest.fixture
    def module(self, tiny_vit_backbone_config, tiny_head_config):
        from src.scdino.models.lightning.dinov2 import DINOv2

        architecture = OmegaConf.create(
            {
                "backbone": tiny_vit_backbone_config,
                "dino_head": tiny_head_config,
                "ibot_head": tiny_head_config,
                "ibot_separate_head": False,
            }
        )
        training = OmegaConf.create(
            {
                "optimizer": {"lr": 1e-3},
                "teacher_temp": {
                    "warmup_steps": 10,
                    "start_value": 0.04,
                    "end_value": 0.07,
                },
                "weight_decay": {"start_value": 0.001, "end_value": 0.01},
                "momentum": {"start_value": 0.992, "end_value": 1.0},
                "losses": {
                    "dino": {
                        "output_dim": 32,
                        "student_temp": 0.1,
                        "center_momentum": 0.9,
                        "center_mode": "mean",
                        "weight": 1.0,
                    },
                    "ibot": {
                        "output_dim": 32,
                        "student_temp": 0.1,
                        "center_momentum": 0.9,
                        "center_mode": "mean",
                        "weight": 1.0,
                    },
                    "koleo": {"p": 2, "eps": 1e-8, "weight": 0.1},
                },
                "knn_eval": {
                    "enable_knn_eval": False,
                    "knn_k": 5,
                    "knn_temperature": 0.07,
                    "knn_max_train_batches": 1,
                    "knn_val_chunk_size": 8,
                    "knn_train_chunk_size": 8,
                },
            }
        )
        return DINOv2(name="dinov2", architecture=architecture, training=training)

    def test_exposes_the_hook(self, module):
        assert hasattr(module, "on_after_backward")

    def test_hook_freezes_the_student_head_in_epoch_zero(self, module, monkeypatch):
        head = module.student_head.dino_head
        head(torch.randn(4, 16)).sum().backward()
        assert any(p.grad is not None for p in head.last_layer.parameters())

        monkeypatch.setattr(type(module), "current_epoch", property(lambda self: 0))
        module.on_after_backward()
        assert all(p.grad is None for p in head.last_layer.parameters())

    def test_hook_releases_the_head_after_warmup(self, module, monkeypatch):
        head = module.student_head.dino_head
        head(torch.randn(4, 16)).sum().backward()

        monkeypatch.setattr(type(module), "current_epoch", property(lambda self: 5))
        module.on_after_backward()
        assert any(p.grad is not None for p in head.last_layer.parameters())

    def test_freeze_window_is_configurable(self, module, monkeypatch):
        module.training_config["freeze_last_layer_epochs"] = 0
        head = module.student_head.dino_head
        head(torch.randn(4, 16)).sum().backward()

        monkeypatch.setattr(type(module), "current_epoch", property(lambda self: 0))
        module.on_after_backward()
        assert any(p.grad is not None for p in head.last_layer.parameters())
