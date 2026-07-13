from typing import Optional

from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from timm.models.vision_transformer import VisionTransformer
from timm.models.resnet import ResNet, BasicBlock

from src.scdino.models.backbones.dinov2 import MaskedVisionTransformerTIMM
try: #rp edit - my uv didn't load these variants - so I skip them if they are not found. 
    from src.scdino.models.backbones.dinov2_StrucPerc import (
        MaskedPerceiverStrucPerc,
        PerceiverStrucPerc,
    )
except ModuleNotFoundError:
    # StrucPerc variant not present in this checkout; only needed if the
    # 'dinov2_StrucPerc' model is selected. Safe to skip for dinov2.
    MaskedPerceiverStrucPerc = None
    PerceiverStrucPerc = None

from src.scdino.models.backbones.dinov3 import (
    DinoVisionTransformer,
    MaskedDinoVisionTransformer,
)
from src.scdino.models.huggingface.configuration_scdino import ScDINOConfig


class ScDINOModel(PreTrainedModel):
    """HuggingFace-compatible wrapper around the ScDINO teacher backbone.

    Supports DINO (ViT / ResNet), DINOv2 (TIMM ``MaskedVisionTransformerTIMM``),
    DINOv2-StrucPerc (Perceiver-style with structured latent grid) and
    DINOv3 (DINOv3 ``DinoVisionTransformer`` wrapped in
    :class:`MaskedDinoVisionTransformer`) backbones for inference.
    """

    config_class = ScDINOConfig

    def __init__(self, config: ScDINOConfig) -> None:
        super().__init__(config)

        if config.model_variant == "dinov2":
            vit = VisionTransformer(
                in_chans=config.in_chans,
                img_size=config.img_size,
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                reg_tokens=config.reg_tokens,
                num_classes=0,
                pos_embed="learn",
                dynamic_img_size=True,
                init_values=1e-5,
            )
            self.backbone = MaskedVisionTransformerTIMM(
                vit=vit,
                antialias=False,
                weight_initialization="skip",
                pos_embed_initialization="skip",
            )

        elif config.model_variant == "dinov3":
            vit = DinoVisionTransformer(
                in_chans=config.in_chans,
                img_size=config.img_size,
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                ffn_ratio=config.mlp_ratio,
                n_storage_tokens=config.reg_tokens,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                ffn_bias=config.ffn_bias,
                mask_k_bias=config.mask_k_bias,
                drop_path_rate=0.0,
                layerscale_init=config.init_values,
                norm_layer=config.norm_layer,
                ffn_layer=config.ffn_layer,
                untie_cls_and_patch_norms=config.untie_cls_and_patch_norms,
                untie_global_and_local_cls_norm=(
                    config.untie_global_and_local_cls_norm
                ),
                pos_embed_rope_base=config.rope_base,
                pos_embed_rope_min_period=config.rope_min_period,
                pos_embed_rope_max_period=config.rope_max_period,
                pos_embed_rope_normalize_coords=config.rope_normalize_coords,
                pos_embed_rope_shift_coords=config.rope_shift_coords,
                pos_embed_rope_jitter_coords=config.rope_jitter_coords,
                pos_embed_rope_rescale_coords=config.rope_rescale_coords,
                pos_embed_rope_dtype=config.rope_dtype,
            )
            vit.init_weights()
            self.backbone = MaskedDinoVisionTransformer(vit=vit)

        elif config.model_variant == "dino":
            if config.backbone_type == "vit":
                self.backbone = VisionTransformer(
                    in_chans=config.in_chans,
                    img_size=config.img_size,
                    patch_size=config.patch_size,
                    embed_dim=config.embed_dim,
                    depth=config.depth,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    reg_tokens=config.reg_tokens,
                    num_classes=0,
                    pos_embed="learn",
                    dynamic_img_size=True,
                    init_values=1e-5,
                )
            elif config.backbone_type == "resnet":
                dim = config.embed_dim
                chs = [
                    max(dim // 8, 32),
                    max(dim // 4, 64),
                    max(dim // 2, 64),
                    dim,
                ]
                self.backbone = ResNet(
                    block=BasicBlock,
                    in_chans=config.in_chans,
                    layers=[2, 2, 2, 2],
                    channels=chs,
                    stem_width=config.stem_width,
                    num_classes=0,
                )
            else:
                raise ValueError(
                    f"Unsupported backbone_type={config.backbone_type!r} "
                    f"for model_variant='dino'"
                )
        else:
            raise ValueError(
                f"Unsupported model_variant={config.model_variant!r}. "
                f"Expected one of 'dino', 'dinov2', 'dinov2_StrucPerc', 'dinov3'."
            )

        self.post_init()

    def forward(
        self,
        pixel_values: Tensor,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.model_variant in (
            "dinov2",
            "dinov2_StrucPerc",
            "dinov3",
        ):
            last_hidden_state = self.backbone.encode(pixel_values)
            pooler_output = last_hidden_state[:, 0]

        elif self.config.backbone_type == "vit":
            last_hidden_state = self.backbone.forward_features(pixel_values)
            pooler_output = last_hidden_state[:, 0]

        else:
            pooler_output = self.backbone(pixel_values)
            last_hidden_state = pooler_output.unsqueeze(1)

        if not return_dict:
            return (last_hidden_state, pooler_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
        )
