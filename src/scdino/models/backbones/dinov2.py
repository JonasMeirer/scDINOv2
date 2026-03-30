from __future__ import annotations

import copy
from functools import partial
from typing import Dict, Any, Optional, Union, Sequence, Tuple, List
import math
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter, Linear, LayerNorm, Identity
from abc import ABC, abstractmethod
from timm.models.vision_transformer import VisionTransformer
from timm.layers.pos_embed import resample_abs_pos_embed

from src.scdino.models import utils


def freeze_eval_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Head(nn.Module):
    def __init__(
        self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead
    ) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2(nn.Module):
    def __init__(
        self,
        backbone_config: Dict[str, Any],
        dino_head_config: Dict[str, Any], 
        ibot_head_config: Dict[str, Any],
        ibot_separate_head: bool = False,
    ) -> None:
        super().__init__()

        # Backbone
        if backbone_config["type"] == "vit":
            backbone_config = backbone_config["vit"]
            teacher = VisionTransformer(
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
        else:
            raise ValueError(f"Invalid backbone type: {backbone_config['type']}")
        
        self.teacher_backbone = MaskedVisionTransformerTIMM(
            vit=teacher,
            antialias=False,
            pos_embed_initialization="skip",
        )
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(
            self.student_backbone.vit,
            drop_path_rate=backbone_config["drop_path_rate"],
            mode="uniform",
        )
        
        freeze_eval_module(self.teacher_backbone)

        # Heads
        dino_head = partial(
            DINOv2ProjectionHead,
            input_dim=dino_head_config["embed_dim"],
            hidden_dim=dino_head_config["hidden_dim"],
            bottleneck_dim=dino_head_config["bottleneck_dim"],
            output_dim=dino_head_config["output_dim"],
            batch_norm=dino_head_config["batch_norm"],
        )

        teacher_dino_head = dino_head()
        student_dino_head = dino_head()

        ibot_head = partial(
            DINOv2ProjectionHead,
            input_dim=ibot_head_config["embed_dim"],
            hidden_dim=ibot_head_config["hidden_dim"],
            bottleneck_dim=ibot_head_config["bottleneck_dim"],
            output_dim=ibot_head_config["output_dim"],
            batch_norm=ibot_head_config["batch_norm"],
        )

        if ibot_separate_head:
            teacher_ibot_head = ibot_head()
            student_ibot_head = ibot_head()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head

        self.teacher_head = DINOv2Head(
            dino_head=teacher_dino_head,
            ibot_head=teacher_ibot_head,
        )
        self.student_head = DINOv2Head(
            dino_head=student_dino_head,
            ibot_head=student_ibot_head,
        )
        
        freeze_eval_module(self.teacher_head)

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher_backbone(x)

    def forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
        self, x: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features
    

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
class DINOv2ProjectionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        batch_norm: bool = False,
    ) -> None:
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
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)  # type: ignore[operator]

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x
    
    
# copy pasted from lightly
# source: https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/masked_vision_transformer.py
class MaskedVisionTransformer(ABC, Module):
    """
    Abstract base class for Masked Vision Transformer models.

    Defines the interface for a Masked Vision Transformer. This class includes abstract
    methods that must be implemented by concrete subclasses to define the forward pass,
    tokenization of images, and various operations needed for the transformer.
    """

    # This is not defined as a property for backwards compatibility.
    # New models should define this as a property.
    mask_token: Parameter

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        ...

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns encoded class tokens from a batch of images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be forwarded.
                Is applied after any masking operation.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tensor with shape (batch_size, embed_dim) containing the encoded class token
            for every image.

        """
        ...

    @abstractmethod
    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Encode input images and return features from the intermediate layers.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be forwarded.
                Is applied after any masking operation.
            norm:
                Apply norm layer to all intermediates.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tuple of batch of encoded output tokens and a list of intermediate features.
            The encoded output tokens have shape (batch_size, embed_dim) and each
            intermediate feature has shape (batch_size, sequence_length, embed_dim).
            If idx_keep is set, only num_tokens_to_keep tokens per sequence are
            returned.
        """
        ...

    @abstractmethod
    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode input images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be encoded.
                Is applied after any masking operation.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            encoded output tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned.
        """
        ...

    def preprocess(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Convert images to tokens, add positional embeddings, and apply masking.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be returned.
                Is applied after any masking operation.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be masked with
                self.mask_token.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            preprocessed tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned. Any class or prefix tokens are prepended to the
            sequence.
        """
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set at the same time.")

        # convert images to tokens
        tokens = self.images_to_tokens(images)
        # add prefix tokens if needed
        tokens = self.prepend_prefix_tokens(tokens)

        if idx_mask is not None:
            tokens = utils.mask_at_index(
                tokens=tokens, index=idx_mask, mask_token=self.mask_token
            )
        elif mask is not None:
            tokens = utils.mask_bool(
                tokens=tokens, mask=mask, mask_token=self.mask_token
            )

        # add positional encoding
        tokens = self.add_pos_embed(tokens)

        if idx_keep is not None:
            tokens = utils.get_at_index(tokens, idx_keep)

        return tokens

    @abstractmethod
    def images_to_tokens(self, images: Tensor) -> Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).

        Returns:
            Tensor with shape (batch_size, num_patches, embed_dim) containing the
            patch tokens (excluding prefix tokens).
        """
        ...

    # Keep for backwards compatibility.
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        return self.prepend_prefix_tokens(x)

    @abstractmethod
    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        """Prepends prefix tokens to the input patch tokens.

        Args:
            x:
                Tensor with shape (batch_size, num_patches, embed_dim) containing patch
                tokens.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing
            the prefix and patch tokens. The prefix tokens are prepended to the
            sequence.
        """
        ...

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        """Adds positional embeddings to the input tokens.

        Args:
            x:
                Tensor with shape (batch_size, sequence_length, embed_dim) containing
                the input tokens. Must include prefix tokens.

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
        """
        ...
    

# copy pasted from lightly
# source: https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/masked_vision_transformer_timm.py
class MaskedVisionTransformerTIMM(MaskedVisionTransformer):
    """Masked Vision Transformer class using TIMM.

    Attributes:
        vit:
            The VisionTransformer object of TIMM.
        mask_token:
            The mask token.
        weight_initialization:
            The weight initialization method. Valid options are ['', 'skip']. '' uses
            the default MAE weight initialization and 'skip' skips the weight
            initialization.
        antialias:
            Whether to use antialiasing when resampling the positional embeddings.
        pos_embed_initialization:
            The strategy to initialize the positional embeddings. Valid options are
            ['learn', 'sincos', 'skip'].

    """

    def __init__(
        self,
        vit: VisionTransformer,
        mask_token: Optional[Parameter] = None,
        weight_initialization: str = "",
        antialias: bool = True,
        pos_embed_initialization: str = "sincos",
    ) -> None:
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token
            if mask_token is not None
            else Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        )

        if weight_initialization not in ("", "skip"):
            raise ValueError(
                f"Invalid weight initialization method: '{weight_initialization}'. "
                "Valid options are: ['', 'skip']."
            )
        if weight_initialization != "skip":
            self._initialize_weights()

        utils.initialize_positional_embedding(
            pos_embedding=self.vit.pos_embed,
            strategy=pos_embed_initialization,
            num_prefix_tokens=self.vit.num_prefix_tokens,
        )

        self.antialias = antialias

    @property
    def sequence_length(self) -> int:
        seq_len: int = self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens
        return seq_len

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask)
        if self.vit.attn_pool is not None:
            x = self.vit.attn_pool(x)
        elif self.vit.global_pool == "avg":
            x = x[:, self.vit.num_prefix_tokens :].mean(dim=1)
        elif self.vit.global_pool:
            x = x[:, 0]  # class token
        return x

    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        # preprocess images, convert to tokens and add positional embeddings
        tokens = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)

        intermediates: List[Tensor] = []
        for blk in self.vit.blocks:
            tokens = blk(tokens)
            intermediates.append(self.vit.norm(tokens) if norm else tokens)

        # normalize
        out: Tensor = self.vit.norm(tokens)

        return out, intermediates

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # preprocess images, convert to tokens and add positional embeddings
        tokens: Tensor = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)
        # apply Transformer blocks
        tokens = self.vit.blocks(tokens)
        # normalize
        tokens = self.vit.norm(tokens)
        return tokens

    def images_to_tokens(self, images: Tensor) -> Tensor:
        tokens: Tensor = self.vit.patch_embed(images)
        if self.vit.dynamic_img_size:
            tokens = tokens.permute(0, 3, 1, 2)  # NHWC -> NCHW
            tokens = tokens.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return tokens

    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        prefix_tokens = []
        if self.vit.cls_token is not None:
            prefix_tokens.append(self.vit.cls_token.expand(x.shape[0], -1, -1))
        if self.vit.reg_token is not None:
            prefix_tokens.append(self.vit.reg_token.expand(x.shape[0], -1, -1))
        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
        x_prefix = x[:, : self.vit.num_prefix_tokens, :]
        x = x[:, self.vit.num_prefix_tokens :, :]
        if self.vit.dynamic_img_size:
            x = x.transpose(1, 2)  # NLC -> NCL
            total_size = torch.numel(x)
            batch_size = x.size(0)
            num_channels = x.size(1)
            grid_size = int(math.sqrt(total_size / (batch_size * num_channels)))
            x = x.view(
                x.size(0),
                x.size(1),
                grid_size,
                grid_size,
            )  # NCL -> NCHW

            # NCHW -> NHWC
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.vit.pos_embed,
                (H, W),
                num_prefix_tokens=(
                    0 if self.vit.no_embed_class else self.vit.num_prefix_tokens
                ),
                antialias=self.antialias,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.vit.pos_embed

        if self.vit.no_embed_class:
            x = x + pos_embed
            if self.vit.num_prefix_tokens:
                x = torch.cat((x_prefix, x), dim=1)
        else:
            if self.vit.num_prefix_tokens:
                x = torch.cat((x_prefix, x), dim=1)
            x = x + pos_embed
        out: Tensor = self.vit.pos_drop(x)
        return out

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.vit.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        if self.vit.has_class_token:
            torch.nn.init.normal_(self.vit.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)


def init_weights(module: Module) -> None:
    if isinstance(module, Linear):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
        

# copy pasted from lightly
# source: https://github.com/lightly-ai/lightly/blob/master/lightly/models/utils.py
def update_drop_path_rate(
    model: VisionTransformer,
    drop_path_rate: float,
    mode: str = "linear",
) -> None:
    """Updates the drop path rate in a TIMM VisionTransformer model.

    Args:
        model:
            TIMM VisionTransformer model.
        drop_path_rate:
            Maximum drop path rate.
        mode:
            Drop path rate update mode. Can be "linear" or "uniform". Linear increases
            the drop path rate from 0 to drop_path_rate over the depth of the model.
            Uniform sets the drop path rate to drop_path_rate for all blocks.
    Raises:
        ValueError: If an unknown mode is provided.
    """
    from timm.layers import DropPath

    total_depth = len(model.blocks)

    # Determine drop path rates based on the specified mode
    if mode == "linear":
        drop_probabilities = np.linspace(0, drop_path_rate, total_depth)
    elif mode == "uniform":
        drop_probabilities = [drop_path_rate for _ in range(total_depth)]
    else:
        raise ValueError(
            f"Unknown mode: '{mode}', supported modes are 'linear' and 'uniform'."
        )

    # Update the drop path rate for each block in the model
    for block, drop_prob in zip(model.blocks, drop_probabilities):
        if drop_prob > 0.0:
            block.drop_path1 = DropPath(drop_prob=drop_path_rate)
            block.drop_path2 = DropPath(drop_prob=drop_path_rate)
        else:
            block.drop_path1 = Identity()
            block.drop_path2 = Identity()

    
    
if __name__ == "__main__":
    # Default config for testing
    backbone_config = {
        "in_chans": 5,
        "img_size": 50,
        "patch_size": 5,
        "embed_dim": 64,
        "depth": 8,
        "num_heads": 8,
        "drop_path_rate": 0.1
    }
    
    dino_head_config = {
        "embed_dim": 64,
        "hidden_dim": 64,
        "bottleneck_dim": 64,
        "output_dim": 64,
        "batch_norm": False
    }
    
    ibot_head_config = {
        "embed_dim": 64,
        "hidden_dim": 64,
        "bottleneck_dim": 64,
        "output_dim": 64,
        "batch_norm": False
    }
    
    model = DINOv2(backbone_config, dino_head_config, ibot_head_config)
    x = torch.randn(1, 5, 50, 50)
    out = model(x)
    print(out.shape)
        
    import code; code.interact(local=dict(globals(), **locals()))
