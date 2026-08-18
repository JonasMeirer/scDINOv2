# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# ---------------------------------------------------------------------------
# PROVENANCE (see THIRD_PARTY_NOTICES.md)
#
# This file has two regions, split by the "scDINO integration" banner further
# down. Everything ABOVE that banner is vendored verbatim from Meta's DINOv3
# reference implementation (https://github.com/facebookresearch/dinov3) and is
# governed by the DINOv3 License Agreement, a copy of which is distributed with
# this repository at licenses/DINOv3-License-Agreement.md. That Agreement is
# NOT the MIT license that covers the rest of scDINO, and it carries
# acceptable-use restrictions -- read it before using or redistributing.
#
# Everything BELOW that banner is original scDINO code under the repository's
# MIT LICENSE. The split is deliberate so the vendored region can be re-synced
# from upstream without merge pain; please keep new code below the banner.
# ---------------------------------------------------------------------------

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Callable

import math
import numpy as np
import torch
import torch.nn.init
from torch import Tensor, nn
import torch.nn.functional as F


def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(
    flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]
) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [
        shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes
    ]
    outputs_reshaped = [
        o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)
    ]
    return outputs_reshaped


def named_replace(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        module = fn(module=module, name=name)
    for child_name_o, child_module in list(module.named_children()):
        child_name = ".".join((name, child_name_o)) if name else child_name_o
        new_child = named_replace(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
        setattr(module, child_name_o, new_child)

    if depth_first and include_root:
        module = fn(module=module, name=name)
    return module


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer(
                "bias_mask", torch.full_like(self.bias, fill_value=math.nan)
            )

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = (
            self.bias * self.bias_mask.to(self.bias.dtype)
            if self.bias is not None
            else None
        )
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(
        self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0,
            is_causal=is_causal,
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(
        rope: tuple[Tensor, Tensor] | None, indices: Tensor
    ) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [
            max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list
        ]
        residual_scale_factors = [
            b / sample_subset_size
            for b, sample_subset_size in zip(b_list, sample_subset_sizes)
        ]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [
                x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)
            ]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1)
                    for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [
                x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)
            ]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = (
            LayerScale(dim, init_values=ls_init_value)
            if ls_init_value
            else nn.Identity()
        )
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(
            dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob
        )

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = (
            LayerScale(dim, init_values=ls_init_value)
            if ls_init_value
            else nn.Identity()
        )

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):

        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


class SwiGLUFFN(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(
            in_features, swiglu_hidden_features, bias=bias, device=device
        )
        self.w2 = nn.Linear(
            in_features, swiglu_hidden_features, bias=bias, device=device
        )
        self.w3 = nn.Linear(
            swiglu_hidden_features, out_features, bias=bias, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        # Alias for compatibility with the rest of the scDINO codebase, which
        # reads ``patch_embed.grid_size`` (TIMM-style).
        self.grid_size = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def sample_augment_state(
        self,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Sample a single (shift, jitter, rescale) augmentation state.

        Returns ``(None, None, None)`` if not training or no augmentation
        is configured. Exposed publicly so that callers needing to reuse
        the same perturbation across multiple ``forward`` calls (e.g. to
        keep the latent-Q and input-KV grids of a cross-attention block
        in a consistent coordinate system) can sample once and pass the
        result to ``forward(..., augment_state=...)``.
        """
        if not self.training:
            return (None, None, None)
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}
        shift_hw: Tensor | None = None
        jitter_hw: Tensor | None = None
        rescale_hw: Tensor | None = None
        if self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(
                -self.shift_coords, self.shift_coords
            )
        if self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_hw = torch.empty(2, **dd).uniform_(-jitter_max, jitter_max).exp()
        if self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_hw = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
        return (shift_hw, jitter_hw, rescale_hw)

    def forward(
        self,
        *,
        H: int,
        W: int,
        augment_state: tuple[Tensor | None, Tensor | None, Tensor | None] | None = None,
    ) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Resolve augmentation state. If the caller provided one we use it
        # as-is (so the same shift/jitter/rescale can be shared across two
        # ``forward`` calls); otherwise sample fresh (matches the original
        # DINOv3 behavior).
        if augment_state is None:
            augment_state = self.sample_augment_state()
        shift_hw, jitter_hw, rescale_hw = augment_state

        # Shift coords by adding a uniform value in [-shift, shift]
        if shift_hw is not None:
            coords = coords + shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if jitter_hw is not None:
            coords = coords * jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if rescale_hw is not None:
            coords = coords * rescale_hw

        # Prepare angles and sin/cos
        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        )  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2
                * torch.arange(self.D_head // 4, device=device, dtype=dtype)
                / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim, device=device)
            )
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(
            f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new"
        )
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(
            f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new"
        )
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    # ------------------------------------------------------------------
    # scDINO compatibility shims
    # ------------------------------------------------------------------

    @property
    def num_prefix_tokens(self) -> int:
        """Total number of non-patch tokens prepended to the sequence
        (CLS + storage tokens). Matches the TIMM ViT attribute that other
        modules in this repo (e.g. ``MaskedVisionTransformerTIMM``) read."""
        return 1 + self.n_storage_tokens

    @property
    def reg_token(self) -> Tensor | None:
        """Alias for ``storage_tokens`` so callers that look for a
        TIMM-style ``reg_token`` attribute keep working."""
        return self.storage_tokens if self.n_storage_tokens > 0 else None

    def prepare_tokens_with_masks(
        self, x: Tensor, masks=None
    ) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(
                        x[:, : self.n_storage_tokens + 1]
                    )
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None
    ) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(
        self, x: Tensor, n: int = 1
    ) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(
                        torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)
                    )
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(
        self, *args, is_training: bool = False, **kwargs
    ) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model


# ---------------------------------------------------------------------------
# scDINO integration: Masked wrapper, top-level DINOv3 model, drop-path helper.
#
# The classes below are intentionally additive on top of the upstream code
# so future syncs from the official DINOv3 repo stay easy. They mirror the
# public surface of ``MaskedVisionTransformerTIMM`` / ``DINOv2`` /
# ``DINOv2StrucPerc`` so the existing Lightning training step, kNN eval,
# and HuggingFace export all work without further edits in those files.
# ---------------------------------------------------------------------------

import copy as _copy  # noqa: E402
from torch.nn import Parameter  # noqa: E402
from scdino.models.backbones.dinov2 import (  # noqa: E402
    DINOv2Head,
    DINOv2ProjectionHead,
    freeze_eval_module,
)


def update_drop_path_rate_dinov3(
    model: DinoVisionTransformer,
    drop_path_rate: float,
    mode: str = "linear",
) -> None:
    """Update the drop-path rate of every block in a ``DinoVisionTransformer``.

    DINOv3 implements drop-path as a per-batch sample skip (controlled by
    ``SelfAttentionBlock.sample_drop_ratio``) rather than the TIMM
    ``DropPath`` module, so this helper only updates the float attribute.
    """
    blocks = list(model.blocks)
    total = len(blocks)
    if mode == "linear":
        probs = np.linspace(0, drop_path_rate, total).tolist()
    elif mode == "uniform":
        probs = [drop_path_rate for _ in range(total)]
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}, supported modes are 'linear' and 'uniform'."
        )
    for blk, p in zip(blocks, probs):
        blk.sample_drop_ratio = float(p)


class MaskedDinoVisionTransformer(nn.Module):
    """Wrapper that exposes a TIMM/lightly-style API around DINOv3.

    Mirrors :class:`scdino.models.backbones.dinov2.MaskedVisionTransformerTIMM`
    closely enough that the Lightning training step, the iBOT masking logic
    and the visualization helpers all work unchanged. The upstream
    :class:`DinoVisionTransformer` is exposed via the ``vit`` property so
    that callers can read ``backbone.vit.patch_embed.grid_size``,
    ``backbone.vit.num_prefix_tokens`` and ``backbone.vit.blocks`` exactly
    like for the TIMM ViT.
    """

    def __init__(
        self,
        vit: DinoVisionTransformer,
        mask_token: Optional[Parameter] = None,
    ) -> None:
        super().__init__()
        self.dinov3_vit = vit
        # DINOv3 ships its own ``mask_token`` (shape ``[1, D]``) inside
        # ``DinoVisionTransformer``. If the caller passes a custom
        # ``mask_token``, replace the one inside the inner ViT so the
        # parameter still lives at exactly one state-dict path
        # (``dinov3_vit.mask_token``) — registering it twice would make
        # HuggingFace ``save_pretrained`` reject the model with a
        # "tied weights" error.
        if mask_token is not None:
            self.dinov3_vit.mask_token = mask_token

    # ``vit`` is a property so that we don't register the same
    # DinoVisionTransformer under two state-dict paths (HuggingFace
    # ``save_pretrained`` rejects tied weights). Same trick as in
    # ``MaskedPerceiverStrucPerc``.
    @property
    def vit(self) -> DinoVisionTransformer:
        return self.dinov3_vit

    @property
    def mask_token(self) -> Parameter:
        return self.dinov3_vit.mask_token

    # ------------------------------------------------------------------
    # Properties / shims
    # ------------------------------------------------------------------

    @property
    def sequence_length(self) -> int:
        return self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens

    def images_to_tokens(self, images: Tensor) -> Tensor:
        """Returns the patch tokens (no prefix, no mask, no RoPE).

        Provided for parity with ``MaskedVisionTransformer.images_to_tokens``.
        Note that the DINOv3 transformer never uses this path internally
        because position information is supplied via RoPE inside each block.
        """
        x = self.vit.patch_embed(images)
        # PatchEmbed returns (B, H, W, D) when flatten_embedding=False.
        if x.ndim == 4:
            x = x.flatten(1, 2)
        return x

    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        prefix = [self.vit.cls_token.expand(x.shape[0], -1, -1)]
        if self.vit.n_storage_tokens > 0:
            prefix.append(self.vit.storage_tokens.expand(x.shape[0], -1, -1))
        return torch.cat(prefix + [x], dim=1)

    def add_pos_embed(self, x: Tensor) -> Tensor:
        # No additive positional embedding: 2D RoPE is applied inside attention.
        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_mask(self, mask: Optional[Tensor]) -> Optional[Tensor]:
        """Convert a ``(B, n_prefix + N_patches)`` mask (the convention used
        by the existing Lightning training step) into the
        ``(B, N_patches)`` form that DINOv3's
        :meth:`DinoVisionTransformer.prepare_tokens_with_masks` expects.

        The prefix portion is required to be all-False (we never mask CLS
        or storage tokens during iBOT) — this matches what
        :func:`scdino.models.lightning.utils.random_block_mask` produces.
        """
        if mask is None:
            return None
        n_prefix = self.vit.num_prefix_tokens
        if mask.shape[1] != self.sequence_length:
            raise ValueError(
                f"mask has shape {tuple(mask.shape)} but expected "
                f"(B, {self.sequence_length})."
            )
        prefix_part = mask[:, :n_prefix]
        if prefix_part.any():
            raise ValueError(
                "MaskedDinoVisionTransformer only supports masks that leave "
                "the prefix tokens (CLS + storage) unmasked."
            )
        return mask[:, n_prefix:]

    # ------------------------------------------------------------------
    # Forward / encode
    # ------------------------------------------------------------------

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the global representation (CLS token).

        Mirrors :meth:`MaskedVisionTransformerTIMM.forward`.
        """
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask)
        return x[:, 0]

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the DINOv3 backbone end-to-end and return the full token
        sequence after the final norm. Shape: ``(B, n_prefix + N_patches, D)``.
        """
        if idx_mask is not None or idx_keep is not None:
            raise NotImplementedError(
                "MaskedDinoVisionTransformer only supports the boolean ``mask`` "
                "argument; ``idx_mask`` / ``idx_keep`` are not implemented."
            )
        patch_mask = self._split_mask(mask)
        x, (H, W) = self.vit.prepare_tokens_with_masks(images, masks=patch_mask)
        for blk in self.vit.blocks:
            rope_sincos = self.vit.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)
        return self.vit.norm(x)

    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        if idx_mask is not None or idx_keep is not None:
            raise NotImplementedError(
                "forward_intermediates does not support idx_mask/idx_keep."
            )
        patch_mask = self._split_mask(mask)
        x, (H, W) = self.vit.prepare_tokens_with_masks(images, masks=patch_mask)
        intermediates: List[Tensor] = []
        for blk in self.vit.blocks:
            rope_sincos = self.vit.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)
            intermediates.append(self.vit.norm(x) if norm else x)
        out = self.vit.norm(x)
        return out, intermediates

    # ------------------------------------------------------------------
    # Attention helpers (visualization)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_last_selfattention(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the softmax attention probs of the **last** block.

        Shape: ``(B, num_heads, N, N)`` with ``N = n_prefix + N_patches``.

        DINOv3's :class:`SelfAttention` uses fused SDPA which never
        materializes the attention matrix, so we manually recompute Q/K
        for the last block (with RoPE) and softmax to recover the weights.
        """
        if idx_mask is not None or idx_keep is not None:
            raise NotImplementedError
        patch_mask = self._split_mask(mask)
        x, (H, W) = self.vit.prepare_tokens_with_masks(images, masks=patch_mask)
        for blk in self.vit.blocks[:-1]:
            rope_sincos = self.vit.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)
        last = self.vit.blocks[-1]
        rope_sincos = self.vit.rope_embed(H=H, W=W)

        x_norm = last.norm1(x)
        a = last.attn
        B_, N, _ = x_norm.shape
        C = a.qkv.in_features
        qkv = a.qkv(x_norm).reshape(B_, N, 3, a.num_heads, C // a.num_heads)
        q, k, _ = torch.unbind(qkv, 2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        if rope_sincos is not None:
            q, k = a.apply_rope(q, k, rope_sincos)
        attn = (q @ k.transpose(-2, -1)) * a.scale
        return attn.softmax(dim=-1)

    @torch.no_grad()
    def get_cls_attention_map(
        self, images: Tensor, head_fusion: str = "mean"
    ) -> Tensor:
        """Return the CLS->patch attention as a spatial heatmap.

        Shapes:
            * ``"mean"`` / ``"max"`` -> ``(B, 1, H, W)``
            * ``"none"`` -> ``(B, num_heads, H, W)``
        """
        attn = self.get_last_selfattention(images)  # (B, h, N, N)
        p = self.vit.num_prefix_tokens
        cls_to_patch = attn[:, :, 0, p:]  # (B, h, H*W)
        B_, h, L = cls_to_patch.shape
        ph, pw = self.vit.patch_embed.patch_size
        H, W = images.shape[-2] // ph, images.shape[-1] // pw
        if H * W != L:
            raise AssertionError(f"{H}*{W} != {L}")
        cls_to_patch = cls_to_patch.reshape(B_, h, H, W)
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


def _build_dinov3_from_config(cfg: Dict[str, Any]) -> DinoVisionTransformer:
    """Instantiate a ``DinoVisionTransformer`` from a dict-like config block."""
    vit = DinoVisionTransformer(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        ffn_ratio=cfg.get("ffn_ratio", cfg.get("mlp_ratio", 4.0)),
        qkv_bias=cfg.get("qkv_bias", True),
        proj_bias=cfg.get("proj_bias", True),
        ffn_bias=cfg.get("ffn_bias", True),
        drop_path_rate=0.0,  # student gets drop path applied separately
        layerscale_init=cfg.get("layerscale_init", cfg.get("init_values", 1e-5)),
        norm_layer=cfg.get("norm_layer", "layernorm"),
        ffn_layer=cfg.get("ffn_layer", "mlp"),
        n_storage_tokens=cfg.get("n_storage_tokens", cfg.get("reg_tokens", 0)),
        mask_k_bias=cfg.get("mask_k_bias", False),
        untie_cls_and_patch_norms=cfg.get("untie_cls_and_patch_norms", False),
        untie_global_and_local_cls_norm=cfg.get(
            "untie_global_and_local_cls_norm", False
        ),
        # 2D RoPE
        pos_embed_rope_base=cfg.get("rope_base", 100.0),
        pos_embed_rope_min_period=cfg.get("rope_min_period", None),
        pos_embed_rope_max_period=cfg.get("rope_max_period", None),
        pos_embed_rope_normalize_coords=cfg.get("rope_normalize_coords", "separate"),
        pos_embed_rope_shift_coords=cfg.get("rope_shift_coords", None),
        pos_embed_rope_jitter_coords=cfg.get("rope_jitter_coords", None),
        pos_embed_rope_rescale_coords=cfg.get("rope_rescale_coords", None),
        pos_embed_rope_dtype=cfg.get("rope_dtype", "fp32"),
    )
    vit.init_weights()
    return vit


class DINOv3(nn.Module):
    """Top-level DINOv3 model with teacher / student backbones + heads.

    Mirrors :class:`scdino.models.backbones.dinov2.DINOv2` and
    :class:`scdino.models.backbones.dinov2_StrucPerc.DINOv2StrucPerc`
    so the same Lightning training loop can drive it. Reuses the existing
    :class:`DINOv2Head` / :class:`DINOv2ProjectionHead` for the projection
    heads.
    """

    def __init__(
        self,
        backbone_config: Dict[str, Any],
        dino_head_config: Dict[str, Any],
        ibot_head_config: Dict[str, Any],
        ibot_separate_head: bool = False,
    ) -> None:
        super().__init__()

        if backbone_config["type"] != "dinov3":
            raise ValueError(
                f"Invalid backbone type: {backbone_config['type']}. "
                "DINOv3 requires backbone_config['type'] == 'dinov3'."
            )
        cfg = backbone_config["dinov3"]

        teacher_vit = _build_dinov3_from_config(cfg)
        self.teacher_backbone = MaskedDinoVisionTransformer(vit=teacher_vit)
        self.student_backbone = _copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate_dinov3(
            self.student_backbone.vit,
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
            mode="uniform",
        )
        freeze_eval_module(self.teacher_backbone)

        # Projection heads (reuse DINOv2 implementations).
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
            dino_head=teacher_dino_head, ibot_head=teacher_ibot_head
        )
        self.student_head = DINOv2Head(
            dino_head=student_dino_head, ibot_head=student_ibot_head
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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    backbone_config = {
        "type": "dinov3",
        "dinov3": {
            "in_chans": 5,
            "img_size": 50,
            "patch_size": 5,
            "embed_dim": 64,
            "depth": 4,
            "num_heads": 8,
            "ffn_ratio": 4.0,
            "n_storage_tokens": 4,
            "drop_path_rate": 0.1,
            "layerscale_init": 1e-5,
            "norm_layer": "layernorm",
            "ffn_layer": "mlp",
            "qkv_bias": True,
            "proj_bias": True,
            "ffn_bias": True,
            "rope_base": 100.0,
            "rope_normalize_coords": "separate",
            "rope_shift_coords": 0.5,
            "rope_jitter_coords": 2.0,
            "rope_rescale_coords": 2.0,
            "rope_dtype": "fp32",
        },
    }
    head_cfg = {
        "embed_dim": 64,
        "hidden_dim": 64,
        "bottleneck_dim": 64,
        "output_dim": 64,
        "batch_norm": False,
    }

    model = DINOv3(backbone_config, head_cfg, head_cfg)
    x = torch.randn(2, 5, 50, 50)

    cls = model(x)
    print("forward (CLS):", cls.shape)

    cls_tokens, features = model.forward_teacher(x)
    print("teacher CLS:", cls_tokens.shape, "features:", features.shape)

    seq_len = model.teacher_backbone.sequence_length
    n_prefix = model.teacher_backbone.vit.num_prefix_tokens
    H, W = model.teacher_backbone.vit.patch_embed.grid_size
    assert features.shape == (2, seq_len, 64), features.shape
    assert seq_len - n_prefix == H * W, (seq_len, n_prefix, H, W)

    # Masked student path (iBOT-style mask on patches only).
    mask = torch.zeros(2, seq_len, dtype=torch.bool)
    mask[:, n_prefix:] = torch.rand(2, seq_len - n_prefix) < 0.4
    cls_s, masked_feats = model.forward_student(x, mask=mask)
    print(
        "student CLS:",
        cls_s.shape,
        "masked feats:",
        masked_feats.shape,
        "(no NaN)",
    )
    assert not torch.isnan(masked_feats).any()

    # Attention helpers.
    attn_self = model.teacher_backbone.get_last_selfattention(x)
    attn_cls = model.teacher_backbone.get_cls_attention_map(x, head_fusion="mean")
    print("last self attn:", attn_self.shape)
    print("cls attn map:", attn_cls.shape)

    # Variable-size input (mimics local crops in DINOv2 multi-crop training).
    x_small = torch.randn(2, 5, 25, 25)
    cls_small = model(x_small)
    print("small forward (CLS):", cls_small.shape)

    # Backward pass to make sure grads flow without NaN.
    loss = cls_s.pow(2).mean() + masked_feats.pow(2).mean()
    loss.backward()
    nans = sum(
        int(torch.isnan(p.grad).any()) for p in model.parameters() if p.grad is not None
    )
    print("NaN-grad params:", nans, "loss:", loss.item())
    assert nans == 0
    print("OK")
