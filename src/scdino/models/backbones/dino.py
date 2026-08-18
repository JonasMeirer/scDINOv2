# PROVENANCE (see THIRD_PARTY_NOTICES.md): the blocks in this file marked
# "copy pasted from lightly" (ProjectionHead, DINOProjectionHead) are vendored
# from Lightly (https://github.com/lightly-ai/lightly), Copyright (c) 2020
# Lightly AG and its affiliates, used under Lightly's MIT license, reproduced
# at licenses/lightly-MIT.txt. The remainder of this file is original scDINO
# code under the repository's MIT LICENSE.

import copy
from typing import Dict, Any, Optional, Union, Sequence, Tuple, List
from warnings import warn
import math

import torch
from torch import nn, Tensor

from timm.models.vision_transformer import VisionTransformer
from timm.models.resnet import ResNet, BasicBlock


def freeze_eval_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


@torch.no_grad()
def vit_get_last_selfattention(vit: nn.Module, images: Tensor) -> Tensor:
    """Return last-block attention probs of a TIMM ``VisionTransformer``.

    Shape: ``(B, num_heads, N, N)``. TIMM's ``Attention.forward`` uses fused
    SDPA which never materializes the softmax matrix, so we run all blocks
    except the last normally and recompute the last block's attention
    explicitly.
    Code adapted from here:
    https://github.com/huggingface/pytorch-image-models/blob/2703d155c88d27bba9a1f465f5489a7947ffc313/timm/models/vision_transformer.py#L698
    """
    if not isinstance(vit, VisionTransformer):
        raise RuntimeError(
            "Attention extraction is only supported for ViT backbones, "
            f"got {type(vit).__name__}."
        )

    x = vit.patch_embed(images)
    x = vit._pos_embed(x)
    x = vit.patch_drop(x)
    x = vit.norm_pre(x)
    for blk in vit.blocks[:-1]:
        x = blk(x)
    last = vit.blocks[-1]
    x = last.norm1(x)
    a = last.attn
    B, N, _ = x.shape
    qkv = (
        a.qkv(x)
        .reshape(B, N, 3, a.num_heads, a.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, _ = qkv.unbind(0)
    q, k = a.q_norm(q), a.k_norm(k)
    attn = (q @ k.transpose(-2, -1)) * a.scale
    return attn.softmax(dim=-1)


@torch.no_grad()
def vit_get_cls_attention_map(
    vit: nn.Module, images: Tensor, head_fusion: str = "mean"
) -> Tensor:
    """Return CLS->patch attention as a spatial heatmap.

    Args:
        vit: A TIMM ``VisionTransformer``.
        images: Tensor of shape ``(B, C, H_img, W_img)``.
        head_fusion: How to combine attention heads. One of
            ``"mean"`` -> ``(B, H, W)``, ``"max"`` -> ``(B, H, W)``,
            ``"none"`` -> ``(B, num_heads, H, W)``.
    """
    attn = vit_get_last_selfattention(vit, images)  # (B, h, N, N)
    p = vit.num_prefix_tokens
    cls_to_patch = attn[:, :, 0, p:]  # (B, h, H*W)
    B, h, L = cls_to_patch.shape
    ph, pw = vit.patch_embed.patch_size
    H, W = images.shape[-2] // ph, images.shape[-1] // pw
    assert H * W == L, f"{H}*{W} != {L}"
    cls_to_patch = cls_to_patch.reshape(B, h, H, W)
    if head_fusion == "mean":
        return cls_to_patch.mean(dim=1).unsqueeze(1)
    if head_fusion == "max":
        return cls_to_patch.amax(dim=1).unsqueeze(1)
    if head_fusion == "none":
        return cls_to_patch
    raise ValueError(
        f"Invalid head_fusion: {head_fusion!r}. "
        "Expected one of: 'mean', 'max', 'none'."
    )


class DINO(nn.Module):
    def __init__(
        self,
        backbone_config: Dict[str, Any],
        dino_head_config: Dict[str, Any],
    ) -> None:
        super().__init__()

        # Backbone
        if backbone_config["type"] == "vit":
            backbone_config = backbone_config["vit"]
            self.teacher_backbone = VisionTransformer(
                in_chans=backbone_config["in_chans"],
                img_size=backbone_config["img_size"],
                patch_size=backbone_config["patch_size"],
                embed_dim=backbone_config["embed_dim"],
                depth=backbone_config["depth"],
                num_heads=backbone_config["num_heads"],
                mlp_ratio=backbone_config["mlp_ratio"],
                reg_tokens=backbone_config["reg_tokens"],
                num_classes=0,
                pos_embed="learn",
                dynamic_img_size=True,
                init_values=1e-5,
            )
        elif backbone_config["type"] == "resnet":
            backbone_config = backbone_config["resnet"]
            dim = backbone_config["embed_dim"]
            chs = [max(dim // 8, 32), max(dim // 4, 64), max(dim // 2, 64), dim]
            self.teacher_backbone = ResNet(
                block=BasicBlock,
                in_chans=backbone_config["in_chans"],
                layers=[2, 2, 2, 2],
                channels=chs,
                stem_width=backbone_config["stem_width"],
                num_classes=0,
            )
        else:
            raise ValueError(f"Invalid backbone type: {backbone_config['type']}")

        self.student_backbone = copy.deepcopy(self.teacher_backbone)

        # Heads
        self.teacher_head = DINOProjectionHead(
            input_dim=dino_head_config["embed_dim"],
            hidden_dim=dino_head_config["hidden_dim"],
            bottleneck_dim=dino_head_config["bottleneck_dim"],
            output_dim=dino_head_config["output_dim"],
            batch_norm=dino_head_config["batch_norm"],
        )
        self.student_head = DINOProjectionHead(
            input_dim=dino_head_config["embed_dim"],
            hidden_dim=dino_head_config["hidden_dim"],
            bottleneck_dim=dino_head_config["bottleneck_dim"],
            output_dim=dino_head_config["output_dim"],
            batch_norm=dino_head_config["batch_norm"],
            freeze_last_layer=1,
        )

        freeze_eval_module(self.teacher_head)
        freeze_eval_module(self.teacher_backbone)

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher_backbone(x)

    def forward_teacher(self, x: Tensor) -> Tensor:
        features = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(features)
        return z

    def forward_student(self, x: Tensor) -> Tensor:
        features = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(features)
        return z

    @torch.no_grad()
    def get_last_selfattention(self, images: Tensor) -> Tensor:
        """Return last-block attention probs of the teacher ViT.

        Shape: ``(B, num_heads, N, N)``. Only supported for ViT backbones.
        See ``vit_get_last_selfattention``.
        """
        return vit_get_last_selfattention(self.teacher_backbone, images)

    @torch.no_grad()
    def get_cls_attention_map(
        self, images: Tensor, head_fusion: str = "mean"
    ) -> Tensor:
        """Return CLS->patch attention as a spatial heatmap.

        See ``vit_get_cls_attention_map``.
        """
        return vit_get_cls_attention_map(
            self.teacher_backbone, images, head_fusion=head_fusion
        )


# copy pasted from lightly
# source: https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/heads.py
class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer, use_bias (optional)).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(
        self,
        blocks: Sequence[
            Union[
                Tuple[int, int, Optional[nn.Module], Optional[nn.Module]],
                Tuple[int, int, Optional[nn.Module], Optional[nn.Module], bool],
            ],
        ],
    ) -> None:
        """Initializes the ProjectionHead module with the specified blocks."""
        super().__init__()

        layers: List[nn.Module] = []
        for block in blocks:
            input_dim, output_dim, batch_norm, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        projection: Tensor = self.layers(x)
        return projection


# copy pasted from lightly
# source: https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/heads.py
class DINOProjectionHead(ProjectionHead):
    """Projection head used in DINO.

    "The projection head consists of a 3-layer multi-layer perceptron (MLP)
    with hidden dimension 2048 followed by l2 normalization and a weight
    normalized fully connected layer with K dimensions, which is similar to the
    design from SwAV [1]." [0]

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: SwAV, 2020, https://arxiv.org/abs/2006.09882

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        output_dim:
            The output dimension of the head.
        batch_norm:
            Whether to use batch norm or not. Should be set to False when using
            a vision transformer backbone.
        freeze_last_layer:
            Number of epochs during which we keep the output layer fixed.
            Typically doing so during the first epoch helps training. Try
            increasing this value if the loss does not decrease.
        norm_last_layer:
            Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the
            training unstable.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        batch_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):
        """Initializes the DINOProjectionHead with the specified dimensions."""
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.GELU(),
                ),
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.GELU(),
                ),
                (hidden_dim, bottleneck_dim, None, None),
            ]
        )
        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        # Tell mypy this is ok because fill_ is overloaded.
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        # Option to normalize last layer.
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel last layer gradients to stabilize the training."""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def forward(self, x: Tensor) -> Tensor:
        """Computes one forward pass through the head."""
        x = self.layers(x)
        # l2 normalization
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution."""
        if isinstance(module, nn.Linear):
            _no_grad_trunc_normal(
                module.weight,
                mean=0,
                std=0.02,
                a=-2,
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


# copy paste from PyTorch master branch as it is not available in older releases
# source: https://github.com/pytorch/pytorch/blob/20ac7362009dd8e0aca6e72fc9357773136a83b8/torch/nn/init.py#L22-L54
def _no_grad_trunc_normal(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Initializes the input tensor with a truncated normal distribution.

    This method is based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Args:
        tensor:
            The tensor to initialize.
        mean:
            Mean of the distribution.
        std:
            Standard deviation of the distribution.
        a:
            Minimum value of the distribution, values below will be clamped.
        b:
            Maximum value of the distribution, values above will be clamped.

    """

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


if __name__ == "__main__":
    # Default config for testing
    backbone_config = {
        "type": "resnet",
        "vit": {
            "in_chans": 5,
            "img_size": 50,
            "patch_size": 5,
            "embed_dim": 128,
            "depth": 8,
            "num_heads": 8,
            "drop_path_rate": 0.1,
        },
        "resnet": {"in_chans": 5, "embed_dim": 128, "stem_width": 32},
    }

    dino_head_config = {
        "embed_dim": 128,
        "hidden_dim": 64,
        "bottleneck_dim": 64,
        "output_dim": 128,
        "batch_norm": False,
    }

    model = DINO(backbone_config, dino_head_config)
    x = torch.randn(3, 5, 50, 50)
    out = model(x)

    print(out.shape)
