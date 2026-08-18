"""Round-trip tests for the HuggingFace export path.

`save_pretrained` rebuilds an `ScDINOModel` from a config it derives from the
training config and then loads the teacher weights into it. If the derived
architecture disagrees with the trained one in any way, the exported checkpoint
is silently wrong. These tests pin the round trip end to end.
"""

import pytest
import torch
from omegaconf import OmegaConf

from src.scdino.models.huggingface import ScDINOConfig, ScDINOModel
from src.scdino.models.lightning.dinov2 import DINOv2


@pytest.fixture
def tiny_dinov2(tiny_vit_backbone_config, tiny_head_config):
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
            "max_epochs": 1,
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


class TestScDINOConfig:
    def test_round_trips_through_a_dict(self):
        cfg = ScDINOConfig(model_variant="dinov2", embed_dim=128, reg_tokens=8)
        restored = ScDINOConfig(**cfg.to_dict())
        assert restored.embed_dim == 128
        assert restored.reg_tokens == 8
        assert restored.model_variant == "dinov2"

    def test_rejects_an_unsupported_variant(self):
        with pytest.raises(ValueError, match="Unsupported model_variant"):
            ScDINOModel(ScDINOConfig(model_variant="not-a-model"))


class TestSavePretrainedRoundTrip:
    def test_reloaded_model_produces_identical_embeddings(self, tiny_dinov2, tmp_path):
        """The whole point of the export: same weights in, same features out."""
        images = torch.randn(2, 5, 16, 16)

        tiny_dinov2.eval()
        with torch.no_grad():
            expected, _ = tiny_dinov2.forward_teacher(images)

        tiny_dinov2.save_pretrained(str(tmp_path / "hf_model"))
        reloaded = ScDINOModel.from_pretrained(str(tmp_path / "hf_model"))
        reloaded.eval()
        with torch.no_grad():
            actual = reloaded(images).pooler_output

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_export_preserves_the_trained_architecture(self, tiny_dinov2, tmp_path):
        """Guards against `save_pretrained` falling back to hardcoded defaults
        that disagree with the config the model was actually built from."""
        vit_cfg = tiny_dinov2.backbone_config.vit

        tiny_dinov2.save_pretrained(str(tmp_path / "hf_model"))
        exported = ScDINOConfig.from_pretrained(str(tmp_path / "hf_model"))

        assert exported.embed_dim == vit_cfg.embed_dim
        assert exported.depth == vit_cfg.depth
        assert exported.num_heads == vit_cfg.num_heads
        assert exported.patch_size == vit_cfg.patch_size
        assert exported.img_size == vit_cfg.img_size
        assert exported.in_chans == vit_cfg.in_chans
        assert exported.reg_tokens == vit_cfg.reg_tokens

    def test_export_writes_a_loadable_directory(self, tiny_dinov2, tmp_path):
        out = tmp_path / "hf_model"
        tiny_dinov2.save_pretrained(str(out))
        assert (out / "config.json").is_file()
        assert any(out.glob("*.safetensors")) or (out / "pytorch_model.bin").is_file()

    def test_exported_weights_match_the_teacher_not_the_student(
        self, tiny_dinov2, tmp_path
    ):
        """Only the EMA teacher is meant to be exported."""
        with torch.no_grad():
            for param in tiny_dinov2.student_backbone.parameters():
                param.add_(1.0)  # make the student clearly different

        tiny_dinov2.save_pretrained(str(tmp_path / "hf_model"))
        reloaded = ScDINOModel.from_pretrained(str(tmp_path / "hf_model"))

        teacher_state = tiny_dinov2.teacher_backbone.state_dict()
        for key, value in reloaded.backbone.state_dict().items():
            torch.testing.assert_close(value, teacher_state[key])


class TestEncodeInterface:
    def test_encode_returns_one_cls_vector_per_image(self, tiny_dinov2):
        embeds = tiny_dinov2.encode(torch.randn(3, 5, 16, 16))
        assert embeds.shape == (3, tiny_dinov2.backbone_config.vit.embed_dim)

    def test_forward_teacher_returns_cls_and_full_token_sequence(self, tiny_dinov2):
        cls_tokens, features = tiny_dinov2.forward_teacher(torch.randn(2, 5, 16, 16))
        assert cls_tokens.shape == (2, 16)
        assert features.shape[0] == 2
        # 4x4 patch grid + 1 CLS + 2 register tokens
        assert features.shape[1] == 16 + 1 + 2
        torch.testing.assert_close(cls_tokens, features[:, 0])

    def test_attention_map_covers_the_patch_grid(self, tiny_dinov2):
        heatmap = tiny_dinov2.get_cls_attention_map(torch.randn(2, 5, 16, 16))
        assert heatmap.shape[0] == 2
        assert heatmap.shape[-2:] == (4, 4)
