"""StrucPerc DINOv2 backbone.

A Perceiver-style vision encoder for the scDINO codebase.

Differences vs. the standard ViT (``dinov2.py``):

* The input image is patch-embedded onto a (potentially large) input grid, e.g.
  ``input_grid = (50, 50)`` with ``patch_size=1`` on a 50x50 image.
* A learnable bank of latent tokens lives on a smaller, configurable
  spatial grid (e.g. ``latent_size = (25, 25)``).
* Cross-attention with axial 2D RoPE on Q/K (V is **not** rotated) is used to
  let the latent queries attend to input keys/values. Both grids are expressed
  in *latent-cell units* so the relative offset between a latent token at
  ``(x_l, y_l)`` and an input token at ``(x_i, y_i)`` has a natural meaning.
* Latent self-attention blocks (also with 2D RoPE on Q/K) refine the latent
  representation. CLS and register tokens are concatenated with the latent
  tokens but RoPE is forced to identity on them (no spatial position).
* iBOT masking lives at the **latent** grid: masked latent positions are
  replaced with ``mask_token`` after the first cross-attention block and
  before the latent self-attention stack. This makes the wrapper a drop-in
  replacement for ``MaskedVisionTransformerTIMM`` w.r.t. the existing
  Lightning training step.

This module exposes:

* ``RoPE2D``, ``MultiHeadAttentionRoPE2D``, ``CrossAttentionBlock``,
  ``SelfAttentionBlock`` (low-level building blocks).
* ``PerceiverPatchEmbed``, ``PerceiverStrucPerc`` (the inner backbone that
  also acts as the ``vit`` namespace expected by the rest of the codebase).
* ``MaskedPerceiverStrucPerc`` (mirrors ``MaskedVisionTransformerTIMM``).
* ``DINOv2StrucPerc`` (mirrors ``DINOv2`` and reuses the existing
  projection-head infrastructure).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Identity, LayerNorm, Linear, Module, Parameter

from timm.layers import DropPath, Mlp

try:
    from timm.models.vision_transformer import LayerScale
except ImportError:  # pragma: no cover - very old timm
    class LayerScale(nn.Module):
        def __init__(
            self, dim: int, init_values: float = 1e-5, inplace: bool = False
        ) -> None:
            super().__init__()
            self.inplace = inplace
            self.gamma = nn.Parameter(init_values * torch.ones(dim))

        def forward(self, x: Tensor) -> Tensor:
            return x.mul_(self.gamma) if self.inplace else x * self.gamma


from src.scdino.models.backbones.dinov2 import (
    DINOv2Head,
    DINOv2ProjectionHead,
    freeze_eval_module,
)


# ---------------------------------------------------------------------------
# 2D RoPE
# ---------------------------------------------------------------------------


_AugmentState = Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]


class RopePositionEmbedding(nn.Module):
    """Axial 2D Rotary Position Embedding (DINOv3-style).

    Closely mirrors the reference implementation from DINOv3's
    ``RopePositionEmbedding``:

    * Stores ``periods`` of shape ``(D_head // 4,)`` either from a single
      ``base`` (then ``periods[k] = base ** (2k / (D_head/2))``) or from an
      explicit ``(min_period, max_period)`` range.
    * Coordinates live in ``[-1, +1]`` after a per-image normalisation
      (``"min"`` / ``"max"`` / ``"separate"``).
    * Per-forward training-time augmentations: ``shift_coords``,
      ``jitter_coords``, ``rescale_coords``.
    * Returns ``(sin, cos)`` of shape ``(H * W, D_head)`` ready to be
      multiplied with ``q`` / ``k`` after expanding over batch and heads.

    The two intentional differences vs. DINOv3:

    * ``forward`` accepts an optional ``augment_state`` so that a single
      sampled augmentation can be **shared** across two calls (e.g. between
      the latent Q grid and the input KV grid in cross-attention).
    * ``head_dim`` is required at construction time (DINOv3 derives it from
      ``embed_dim`` / ``num_heads``); same constraint, just expressed at the
      level we instantiate the module.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"RopePositionEmbedding requires head_dim divisible by 4, "
                f"got {head_dim}."
            )
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (
            base is not None and both_periods
        ):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided "
                "(but not both)."
            )

        self.head_dim = head_dim
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        # ``dtype`` is stored separately because ``self.periods.dtype`` may
        # differ once ``.to(...)`` is called by the user.
        self.dtype = dtype

        # Persistent so HuggingFace ``from_pretrained`` (which uses meta init
        # and only restores tensors present in the saved state dict) keeps
        # this buffer initialised, and so that
        # ``teacher.load_state_dict(student.state_dict())`` works.
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype
        Dq = self.head_dim // 4
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(Dq, device=device, dtype=dtype) / (self.head_dim // 2)
            )
        else:
            assert self.min_period is not None and self.max_period is not None
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, Dq, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def sample_augment_state(
        self, device: torch.device, dtype: Optional[torch.dtype] = None
    ) -> _AugmentState:
        """Sample one ``(shift, jitter, rescale)`` triplet for this forward.

        Returns ``(None, None, None)`` if not training or no augmentation
        is configured.
        """
        if not self.training:
            return (None, None, None)
        dt = dtype if dtype is not None else self.dtype
        shift_hw: Optional[Tensor] = None
        jitter_hw: Optional[Tensor] = None
        rescale_hw: Optional[Tensor] = None
        if self.shift_coords is not None:
            shift_hw = torch.empty(2, device=device, dtype=dt).uniform_(
                -self.shift_coords, self.shift_coords
            )
        if self.jitter_coords is not None:
            jmax = math.log(self.jitter_coords)
            jitter_hw = (
                torch.empty(2, device=device, dtype=dt).uniform_(-jmax, jmax).exp()
            )
        if self.rescale_coords is not None:
            rmax = math.log(self.rescale_coords)
            rescale_hw = (
                torch.empty(1, device=device, dtype=dt).uniform_(-rmax, rmax).exp()
            )
        return (shift_hw, jitter_hw, rescale_hw)

    def _coords(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: Optional[torch.dtype],
        augment_state: _AugmentState,
    ) -> Tensor:
        dd = {"device": device, "dtype": dtype}
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(
                f"Unknown normalize_coords: {self.normalize_coords!r}."
            )
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )  # (H, W, 2)
        coords = coords.flatten(0, 1)  # (HW, 2)
        coords = 2.0 * coords - 1.0  # [-1, +1]

        shift_hw, jitter_hw, rescale_hw = augment_state
        if shift_hw is not None:
            coords = coords + shift_hw[None, :]
        if jitter_hw is not None:
            coords = coords * jitter_hw[None, :]
        if rescale_hw is not None:
            coords = coords * rescale_hw
        return coords

    def _coords_to_sincos(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        # coords: (HW, 2). periods: (D_head // 4,).
        periods = self.periods.to(dtype=coords.dtype, device=coords.device)
        angles = (
            2 * math.pi * coords[:, :, None] / periods[None, None, :]
        )  # (HW, 2, D//4)
        angles = angles.flatten(1, 2)  # (HW, D//2)
        angles = angles.tile(2)  # (HW, D)
        return angles.sin(), angles.cos()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        H: int,
        W: int,
        augment_state: Optional[_AugmentState] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute ``(sin, cos)`` of shape ``(H * W, D_head)`` for an
        ``H x W`` regular grid.

        If ``augment_state`` is provided it is used as-is so the same
        translation / jitter / rescale can be shared between Q and KV in
        cross-attention. Otherwise a fresh state is sampled (training only).
        """
        device = self.periods.device
        dtype = self.dtype
        if augment_state is None:
            augment_state = self.sample_augment_state(device, dtype)
        coords = self._coords(H, W, device, dtype, augment_state)
        return self._coords_to_sincos(coords)


def apply_rotary(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply rotary position embedding.

    Args:
        x: ``(B, h, N, D)``.
        sin, cos: ``(N, D)`` (will broadcast over batch and heads) or any
            broadcast-compatible shape.

    Returns:
        Tensor of shape ``(B, h, N, D)``.
    """
    # rotate_half (LLaMA convention): cat(-x2, x1) where x1, x2 = chunk(2).
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    # Make sin/cos broadcastable against (B, h, N, D).
    while sin.dim() < x.dim():
        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)
    return x * cos + rotated * sin


def pad_sincos_for_prefix(
    sin: Tensor, cos: Tensor, n_prefix: int
) -> Tuple[Tensor, Tensor]:
    """Prepend identity rotations (sin=0, cos=1) for ``n_prefix`` tokens.

    Used for CLS / register tokens that should not have any spatial RoPE
    applied to them.
    """
    if n_prefix <= 0:
        return sin, cos
    pad_sin = sin.new_zeros(n_prefix, sin.shape[-1])
    pad_cos = cos.new_ones(n_prefix, cos.shape[-1])
    sin = torch.cat([pad_sin, sin], dim=0)
    cos = torch.cat([pad_cos, cos], dim=0)
    return sin, cos


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class MultiHeadAttentionRoPE2D(nn.Module):
    """Multi-head attention with 2D RoPE on Q and K (never on V).

    Pre-computed ``(sin, cos)`` pairs are passed in by the caller — this
    keeps the shared :class:`RopePositionEmbedding` instance out of every
    attention block's submodule tree (so the buffer ``periods`` lives only
    once in the state dict, under the parent Perceiver).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop_p = float(attn_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q_tokens: Tensor,
        kv_tokens: Tensor,
        q_sincos: Tuple[Tensor, Tensor],
        kv_sincos: Tuple[Tensor, Tensor],
        return_attn: bool = False,
    ) -> Tensor:
        B, Nq, C = q_tokens.shape
        Nkv = kv_tokens.shape[1]
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(q_tokens).view(B, Nq, h, d).permute(0, 2, 1, 3)
        k = self.k_proj(kv_tokens).view(B, Nkv, h, d).permute(0, 2, 1, 3)
        v = self.v_proj(kv_tokens).view(B, Nkv, h, d).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q_sin, q_cos = q_sincos
        k_sin, k_cos = kv_sincos
        q = apply_rotary(q, q_sin.to(dtype=q.dtype), q_cos.to(dtype=q.dtype))
        k = apply_rotary(k, k_sin.to(dtype=k.dtype), k_cos.to(dtype=k.dtype))

        if return_attn:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            return attn.softmax(dim=-1)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0
        )
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block: latent Q attends to input K, V."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = 1e-5,
        qk_norm: bool = False,
        act_layer: type = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1_q = LayerNorm(dim)
        self.norm1_kv = LayerNorm(dim)
        self.attn = MultiHeadAttentionRoPE2D(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else Identity()

        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else Identity()

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        q_sincos: Tuple[Tensor, Tensor],
        kv_sincos: Tuple[Tensor, Tensor],
        return_attn: bool = False,
    ) -> Tensor:
        if return_attn:
            return self.attn(
                self.norm1_q(q),
                self.norm1_kv(kv),
                q_sincos,
                kv_sincos,
                return_attn=True,
            )
        h = self.attn(self.norm1_q(q), self.norm1_kv(kv), q_sincos, kv_sincos)
        q = q + self.drop_path1(self.ls1(h))
        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block on the latent tokens (with prefix)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = 1e-5,
        qk_norm: bool = False,
        act_layer: type = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttentionRoPE2D(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else Identity()

        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else Identity()

    def forward(
        self,
        x: Tensor,
        sincos: Tuple[Tensor, Tensor],
        return_attn: bool = False,
    ) -> Tensor:
        x_norm = self.norm1(x)
        if return_attn:
            return self.attn(x_norm, x_norm, sincos, sincos, return_attn=True)
        h = self.attn(x_norm, x_norm, sincos, sincos)
        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Patch embedding + Perceiver
# ---------------------------------------------------------------------------


class PerceiverPatchEmbed(nn.Module):
    """Conv2d patch embedding for the Perceiver input branch.

    Note:
        For compatibility with the existing Lightning training loop, the
        ``grid_size`` and ``num_patches`` attributes report the **latent**
        grid (not the input grid). The true input grid is exposed through
        ``input_grid_size`` and ``num_input_patches``.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_size: int,
        img_size: int,
        latent_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        ps = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        isz = (img_size, img_size) if isinstance(img_size, int) else img_size
        if isz[0] % ps[0] != 0 or isz[1] % ps[1] != 0:
            raise ValueError(
                f"img_size {isz} must be divisible by patch_size {ps}."
            )
        self.patch_size = ps
        self.img_size = isz
        self.input_grid_size = (isz[0] // ps[0], isz[1] // ps[1])
        self.num_input_patches = self.input_grid_size[0] * self.input_grid_size[1]

        # Reported sizes are the latent grid (drives iBOT masking + sequence_length).
        self.grid_size = tuple(latent_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=ps, stride=ps)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.proj(x)  # (B, D, H_in, W_in)
        h_in, w_in = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, N_in, D)
        return x, (h_in, w_in)


class PerceiverStrucPerc(nn.Module):
    """Perceiver-style backbone with a spatially structured latent grid.

    This module is also used as the ``vit`` namespace by
    ``MaskedPerceiverStrucPerc`` so that the existing Lightning code paths
    that reach into ``backbone.vit.patch_embed.grid_size``,
    ``backbone.vit.num_prefix_tokens`` and ``backbone.vit.blocks`` continue
    to work without modification.
    """

    def __init__(
        self,
        in_chans: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        latent_size: Tuple[int, int] = (25, 25),
        num_self_blocks: int = 8,
        num_cross_blocks: int = 1,
        depth_outer: int = 1,
        reg_tokens: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 1e-5,
        # 2D RoPE (DINOv3-style)
        rope_base: Optional[float] = 100.0,
        rope_min_period: Optional[float] = None,
        rope_max_period: Optional[float] = None,
        rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        rope_shift_coords: Optional[float] = None,
        rope_jitter_coords: Optional[float] = None,
        rope_rescale_coords: Optional[float] = None,
    ) -> None:
        super().__init__()
        if depth_outer < 1:
            raise ValueError("depth_outer must be >= 1.")
        if num_cross_blocks < 1:
            raise ValueError("num_cross_blocks must be >= 1.")
        if num_self_blocks < 0:
            raise ValueError("num_self_blocks must be >= 0.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_size = tuple(latent_size)
        self.latent_h, self.latent_w = self.latent_size
        self.L = self.latent_h * self.latent_w
        self.reg_tokens_count = reg_tokens
        self.has_class_token = True
        self.num_prefix_tokens = 1 + reg_tokens

        self.depth_outer = int(depth_outer)
        self.num_cross_blocks = int(num_cross_blocks)
        self.num_self_blocks = int(num_self_blocks)

        # Compatibility shims for the existing MaskedVisionTransformerTIMM/
        # Lightning code paths that probe attributes on ``vit``.
        self.dynamic_img_size = False
        self.no_embed_class = True
        self.global_pool = ""
        self.attn_pool = None
        # ``pos_embed`` is queried by ``MaskedVisionTransformerTIMM`` in
        # ``add_pos_embed``; we never use it but keep an empty parameter so
        # the attribute exists.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)

        # Patch embed (input branch).
        self.patch_embed = PerceiverPatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            latent_size=self.latent_size,
        )

        # Learnable latent token bank.
        self.latent_tokens = nn.Parameter(torch.zeros(1, self.L, embed_dim))
        nn.init.trunc_normal_(self.latent_tokens, std=0.02)

        # CLS + register tokens.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if reg_tokens > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim))
            nn.init.trunc_normal_(self.reg_token, std=0.02)
        else:
            self.reg_token = None

        # 2D RoPE (shared across all attention layers; only the parent
        # Perceiver owns the ``periods`` buffer).
        self.rope = RopePositionEmbedding(
            head_dim=self.head_dim,
            base=rope_base,
            min_period=rope_min_period,
            max_period=rope_max_period,
            normalize_coords=rope_normalize_coords,
            shift_coords=rope_shift_coords,
            jitter_coords=rope_jitter_coords,
            rescale_coords=rope_rescale_coords,
        )

        # Drop path schedule (uniform across cross + self blocks initially).
        total_blocks = self.num_cross_blocks * self.depth_outer + self.num_self_blocks
        dp_rates = [drop_path_rate for _ in range(max(total_blocks, 1))]

        # Cross-attention blocks.
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dp_rates[i],
                    init_values=init_values,
                    qk_norm=qk_norm,
                )
                for i in range(self.num_cross_blocks * self.depth_outer)
            ]
        )

        # Self-attention blocks (latent + prefix).
        offset = self.num_cross_blocks * self.depth_outer
        self.self_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dp_rates[offset + i],
                    init_values=init_values,
                    qk_norm=qk_norm,
                )
                for i in range(self.num_self_blocks)
            ]
        )

        # Final norm and (no-op) pre-norm/dropout for compat.
        self.norm = LayerNorm(embed_dim)
        self.norm_pre = nn.Identity()
        self.pos_drop = nn.Identity()

        self.apply(_init_weights)

    # Compatibility alias used by code that iterates ``vit.blocks`` (e.g.
    # ``update_drop_path_rate`` and ``get_last_selfattention``). Implemented
    # as a property so it does not create a duplicate submodule path in the
    # state dict (HuggingFace safetensors refuses tied weights).
    @property
    def blocks(self) -> nn.ModuleList:
        return self.self_blocks

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------

    def _compute_q_kv_sincos(
        self, h_in: int, w_in: int
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Compute matched ``(sin, cos)`` for the latent and input grids.

        A single augmentation state is sampled and reused for both grids so
        that the relative spatial geometry between latent queries and input
        keys/values is preserved (otherwise ``shift`` / ``jitter`` /
        ``rescale`` would create independent perturbations).
        """
        device = self.rope.periods.device
        dtype = self.rope.dtype
        augment_state = self.rope.sample_augment_state(device, dtype)

        q_sin, q_cos = self.rope(
            H=self.latent_h, W=self.latent_w, augment_state=augment_state
        )
        # Identity rotation for prefix (CLS + register) tokens.
        q_sin, q_cos = pad_sincos_for_prefix(q_sin, q_cos, self.num_prefix_tokens)

        kv_sin, kv_cos = self.rope(H=h_in, W=w_in, augment_state=augment_state)
        return (q_sin, q_cos), (kv_sin, kv_cos)

    def _compute_self_sincos(
        self, augment_state: Optional[_AugmentState] = None
    ) -> Tuple[Tensor, Tensor]:
        """``(sin, cos)`` for the latent grid (with prefix padding).

        Used by latent self-attention. Reuses an augmentation state if
        provided so the latent self-attention shares the same perturbation
        as the cross-attention before it.
        """
        sin, cos = self.rope(
            H=self.latent_h, W=self.latent_w, augment_state=augment_state
        )
        return pad_sincos_for_prefix(sin, cos, self.num_prefix_tokens)

    def build_initial_queries(self, batch_size: int) -> Tensor:
        """Construct the initial (prefix + latent) query tokens."""
        latent = self.latent_tokens.expand(batch_size, -1, -1)
        prefix = [self.cls_token.expand(batch_size, -1, -1)]
        if self.reg_token is not None:
            prefix.append(self.reg_token.expand(batch_size, -1, -1))
        prefix_t = torch.cat(prefix, dim=1)
        return torch.cat([prefix_t, latent], dim=1)

    def encode(
        self,
        images: Tensor,
        mask: Optional[Tensor] = None,
        mask_token: Optional[Tensor] = None,
        return_intermediates: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """Run the Perceiver pipeline and return the final token sequence.

        Args:
            images: ``(B, C, H, W)``.
            mask: Optional ``(B, n_prefix + L)`` boolean mask. Tokens where
                ``mask`` is True are replaced by ``mask_token`` after the
                first cross-attention block.
            mask_token: ``(1, 1, D)`` parameter to substitute at masked
                positions. Required if ``mask`` is provided.
            return_intermediates: If True, returns a list of all
                intermediate self-attention outputs.

        Returns:
            ``(B, n_prefix + L, D)`` final tokens (after final ``norm``).
            If ``return_intermediates`` is True, also returns the list of
            per-block outputs.
        """
        if mask is not None and mask_token is None:
            raise ValueError("mask_token must be provided when mask is not None.")

        B = images.shape[0]

        in_tokens, (h_in, w_in) = self.patch_embed(images)  # (B, N_in, D)
        q = self.build_initial_queries(B)  # (B, n_prefix + L, D)
        q_sincos, kv_sincos = self._compute_q_kv_sincos(h_in, w_in)

        intermediates: List[Tensor] = []
        mask_applied = False

        if self.depth_outer == 1:
            for blk in self.cross_blocks:
                q = blk(q, in_tokens, q_sincos, kv_sincos)
            if mask is not None:
                q = _apply_mask_token(q, mask, mask_token)
                mask_applied = True
            for blk in self.self_blocks:
                q = blk(q, q_sincos)
                if return_intermediates:
                    intermediates.append(q)
        else:
            # Iterated mode: depth_outer x (cross blocks then a slice of self
            # blocks). Mask is applied once after the very first cross block.
            self_per_outer, remainder = divmod(
                self.num_self_blocks, self.depth_outer
            )
            self_idx = 0
            cross_idx = 0
            for outer in range(self.depth_outer):
                for _ in range(self.num_cross_blocks):
                    q = self.cross_blocks[cross_idx](
                        q, in_tokens, q_sincos, kv_sincos
                    )
                    cross_idx += 1
                    if mask is not None and not mask_applied:
                        q = _apply_mask_token(q, mask, mask_token)
                        mask_applied = True
                this_outer = self_per_outer + (1 if outer < remainder else 0)
                for _ in range(this_outer):
                    q = self.self_blocks[self_idx](q, q_sincos)
                    if return_intermediates:
                        intermediates.append(q)
                    self_idx += 1

        q = self.norm(q)

        if return_intermediates:
            return q, intermediates
        return q

    @torch.no_grad()
    def last_selfattention(self, images: Tensor) -> Tensor:
        """Return softmax attention probs of the last self-attention block."""
        B = images.shape[0]

        if len(self.self_blocks) == 0:
            raise RuntimeError(
                "last_selfattention requires at least one self-attention block."
            )

        in_tokens, (h_in, w_in) = self.patch_embed(images)
        q = self.build_initial_queries(B)
        q_sincos, kv_sincos = self._compute_q_kv_sincos(h_in, w_in)

        if self.depth_outer == 1:
            for blk in self.cross_blocks:
                q = blk(q, in_tokens, q_sincos, kv_sincos)
            for blk in self.self_blocks[:-1]:
                q = blk(q, q_sincos)
            return self.self_blocks[-1](q, q_sincos, return_attn=True)

        # Iterated mode: replay the full pipeline up to the last self block.
        self_per_outer, remainder = divmod(self.num_self_blocks, self.depth_outer)
        self_idx = 0
        cross_idx = 0
        last_block = self.self_blocks[-1]
        for outer in range(self.depth_outer):
            for _ in range(self.num_cross_blocks):
                q = self.cross_blocks[cross_idx](q, in_tokens, q_sincos, kv_sincos)
                cross_idx += 1
            this_outer = self_per_outer + (1 if outer < remainder else 0)
            for k in range(this_outer):
                blk = self.self_blocks[self_idx]
                self_idx += 1
                if blk is last_block:
                    return blk(q, q_sincos, return_attn=True)
                q = blk(q, q_sincos)
        raise RuntimeError("Unreachable: failed to reach last self-attention block.")

    @torch.no_grad()
    def first_cross_attention(self, images: Tensor) -> Tensor:
        """Return softmax probs of the first cross-attention block.

        Shape: ``(B, h, n_prefix + L, N_in)``.
        """
        B = images.shape[0]
        in_tokens, (h_in, w_in) = self.patch_embed(images)
        q = self.build_initial_queries(B)
        q_sincos, kv_sincos = self._compute_q_kv_sincos(h_in, w_in)
        return self.cross_blocks[0](
            q, in_tokens, q_sincos, kv_sincos, return_attn=True
        )


def _apply_mask_token(tokens: Tensor, mask: Tensor, mask_token: Tensor) -> Tensor:
    """Replace tokens at True positions of ``mask`` with ``mask_token``."""
    m = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    return tokens * (1.0 - m) + m * mask_token


def _init_weights(module: Module) -> None:
    if isinstance(module, Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


# ---------------------------------------------------------------------------
# Wrapper (mirrors MaskedVisionTransformerTIMM)
# ---------------------------------------------------------------------------


class MaskedPerceiverStrucPerc(nn.Module):
    """Drop-in replacement for ``MaskedVisionTransformerTIMM``.

    Provides ``encode``, ``forward``, ``forward_intermediates``,
    ``get_last_selfattention``, ``get_cls_attention_map``, the
    ``sequence_length`` property and the ``vit`` namespace expected by the
    rest of the codebase.
    """

    mask_token: Parameter

    def __init__(
        self,
        perceiver: PerceiverStrucPerc,
        mask_token: Optional[Parameter] = None,
    ) -> None:
        super().__init__()
        self.perceiver = perceiver
        self.mask_token = (
            mask_token
            if mask_token is not None
            else Parameter(torch.zeros(1, 1, perceiver.embed_dim))
        )

    # Expose ``vit`` so existing code that reads
    # ``backbone.vit.patch_embed.grid_size`` etc. works unchanged. Implemented
    # as a property so it does not duplicate submodules in the state dict.
    @property
    def vit(self) -> PerceiverStrucPerc:
        return self.perceiver

    # ------------------------------------------------------------------
    # Properties / abstract method shims
    # ------------------------------------------------------------------

    @property
    def sequence_length(self) -> int:
        return self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens

    def images_to_tokens(self, images: Tensor) -> Tensor:
        """Return the input branch's patch tokens (not used in encode)."""
        tokens, _ = self.vit.patch_embed(images)
        return tokens

    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        prefix = [self.vit.cls_token.expand(x.shape[0], -1, -1)]
        if self.vit.reg_token is not None:
            prefix.append(self.vit.reg_token.expand(x.shape[0], -1, -1))
        return torch.cat(prefix + [x], dim=1)

    def add_pos_embed(self, x: Tensor) -> Tensor:
        # No global position embedding: 2D RoPE is applied inside attention.
        return x

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
        # Mirrors MaskedVisionTransformerTIMM.forward: returns the global
        # representation (CLS by default).
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask)
        return x[:, 0]

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if idx_mask is not None:
            raise NotImplementedError(
                "MaskedPerceiverStrucPerc only supports the boolean ``mask`` "
                "argument; ``idx_mask`` is not implemented."
            )
        if idx_keep is not None:
            raise NotImplementedError(
                "MaskedPerceiverStrucPerc does not support ``idx_keep``."
            )
        return self.perceiver.encode(images, mask=mask, mask_token=self.mask_token)

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
        out, intermediates = self.perceiver.encode(
            images,
            mask=mask,
            mask_token=self.mask_token,
            return_intermediates=True,
        )
        if norm:
            intermediates = [self.vit.norm(t) for t in intermediates]
        return out, intermediates

    @torch.no_grad()
    def get_last_selfattention(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return softmax attention probs of the last latent self-attention block.

        Shape: ``(B, num_heads, n_prefix + L, n_prefix + L)``.
        """
        return self.perceiver.last_selfattention(images)

    @torch.no_grad()
    def get_cls_attention_map(
        self, images: Tensor, head_fusion: str = "mean"
    ) -> Tensor:
        """Return the CLS query's cross-attention over input tokens.

        For a Perceiver this is the natural CLS->image heatmap. The result
        is reshaped to the **input** patch grid (the spatial grid the user
        actually sees in the image), with the same ``head_fusion`` semantics
        as ``MaskedVisionTransformerTIMM.get_cls_attention_map``.

        Shape:
            * ``"mean"`` -> ``(B, 1, H_in, W_in)``
            * ``"max"`` -> ``(B, 1, H_in, W_in)``
            * ``"none"`` -> ``(B, num_heads, H_in, W_in)``
        """
        attn = self.perceiver.first_cross_attention(images)  # (B, h, P+L, N_in)
        cls_to_input = attn[:, :, 0, :]  # (B, h, N_in)
        # Compute the actual input grid from the image (handles crop sizes
        # that differ from the configured ``img_size``).
        ph, pw = self.vit.patch_embed.patch_size
        H_in = images.shape[-2] // ph
        W_in = images.shape[-1] // pw
        B, h, L_in = cls_to_input.shape
        if H_in * W_in != L_in:
            raise AssertionError(f"{H_in}*{W_in} != {L_in}")
        cls_to_input = cls_to_input.reshape(B, h, H_in, W_in)
        if head_fusion == "mean":
            return cls_to_input.mean(dim=1).unsqueeze(1)
        if head_fusion == "max":
            return cls_to_input.amax(dim=1).unsqueeze(1)
        if head_fusion == "none":
            return cls_to_input
        raise ValueError(
            f"Invalid head_fusion: {head_fusion!r}. "
            "Expected one of: 'mean', 'max', 'none'."
        )


# ---------------------------------------------------------------------------
# Drop-path scheduling helper (covers both cross + self blocks)
# ---------------------------------------------------------------------------


def update_drop_path_rate(
    perceiver: PerceiverStrucPerc,
    drop_path_rate: float,
    mode: str = "linear",
) -> None:
    """Update the drop-path rate of all cross + self attention blocks."""
    blocks = list(perceiver.cross_blocks) + list(perceiver.self_blocks)
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
        if p > 0.0:
            blk.drop_path1 = DropPath(drop_prob=p)
            blk.drop_path2 = DropPath(drop_prob=p)
        else:
            blk.drop_path1 = Identity()
            blk.drop_path2 = Identity()


# ---------------------------------------------------------------------------
# Top-level model (mirrors DINOv2 in dinov2.py)
# ---------------------------------------------------------------------------


def _build_perceiver_from_config(cfg: Dict[str, Any]) -> PerceiverStrucPerc:
    """Instantiate a ``PerceiverStrucPerc`` from a dict-like config block."""
    latent_size = tuple(cfg.get("latent_size", (25, 25)))
    return PerceiverStrucPerc(
        in_chans=cfg["in_chans"],
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        latent_size=latent_size,
        num_self_blocks=cfg.get("num_self_blocks", 8),
        num_cross_blocks=cfg.get("num_cross_blocks", 1),
        depth_outer=cfg.get("depth_outer", 1),
        reg_tokens=cfg.get("reg_tokens", 0),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        qkv_bias=cfg.get("qkv_bias", True),
        qk_norm=cfg.get("qk_norm", False),
        drop_rate=cfg.get("drop_rate", 0.0),
        attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=0.0,  # student gets drop path applied separately
        init_values=cfg.get("init_values", 1e-5),
        rope_base=cfg.get("rope_base", 100.0),
        rope_min_period=cfg.get("rope_min_period", None),
        rope_max_period=cfg.get("rope_max_period", None),
        rope_normalize_coords=cfg.get("rope_normalize_coords", "separate"),
        rope_shift_coords=cfg.get("rope_shift_coords", None),
        rope_jitter_coords=cfg.get("rope_jitter_coords", None),
        rope_rescale_coords=cfg.get("rope_rescale_coords", None),
    )


class DINOv2StrucPerc(nn.Module):
    """Top-level DINOv2 model with a StrucPerc backbone.

    Mirrors ``DINOv2`` from ``dinov2.py``: builds a teacher / student pair of
    backbones and projection heads, freezes the teacher, and applies the
    drop-path schedule to the student.
    """

    def __init__(
        self,
        backbone_config: Dict[str, Any],
        dino_head_config: Dict[str, Any],
        ibot_head_config: Dict[str, Any],
        ibot_separate_head: bool = False,
    ) -> None:
        super().__init__()

        if backbone_config["type"] != "perceiver":
            raise ValueError(
                f"Invalid backbone type: {backbone_config['type']}. "
                "DINOv2StrucPerc requires backbone_config['type'] == 'perceiver'."
            )
        cfg = backbone_config["perceiver"]
        teacher_perceiver = _build_perceiver_from_config(cfg)
        self.teacher_backbone = MaskedPerceiverStrucPerc(perceiver=teacher_perceiver)
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        # Apply drop path to the student only (teacher is frozen).
        update_drop_path_rate(
            self.student_backbone.vit,
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
            mode="uniform",
        )

        freeze_eval_module(self.teacher_backbone)

        # Projection heads (reuse existing implementations).
        from functools import partial

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
        "type": "perceiver",
        "perceiver": {
            "in_chans": 5,
            "img_size": 50,
            "patch_size": 2,
            "embed_dim": 64,
            "num_heads": 8,
            "latent_size": (10, 10),
            "num_self_blocks": 4,
            "num_cross_blocks": 1,
            "depth_outer": 1,
            "reg_tokens": 4,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
            "rope_base": 100.0,
            "rope_normalize_coords": "separate",
            "rope_shift_coords": 0.5,
            "rope_jitter_coords": 2.0,
            "rope_rescale_coords": 2.0,
        },
    }
    dino_head_config = {
        "embed_dim": 64,
        "hidden_dim": 64,
        "bottleneck_dim": 64,
        "output_dim": 64,
        "batch_norm": False,
    }
    ibot_head_config = dict(dino_head_config)

    model = DINOv2StrucPerc(backbone_config, dino_head_config, ibot_head_config)
    x = torch.randn(2, 5, 50, 50)

    cls = model(x)
    print("forward (CLS):", cls.shape)

    cls_tokens, features = model.forward_teacher(x)
    print("teacher CLS:", cls_tokens.shape, "features:", features.shape)

    seq_len = model.teacher_backbone.sequence_length
    n_prefix = model.teacher_backbone.vit.num_prefix_tokens
    H, W = model.teacher_backbone.vit.patch_embed.grid_size
    assert features.shape == (2, seq_len, 64), features.shape
    assert seq_len - n_prefix == H * W

    # Masked student path
    mask = torch.zeros(2, seq_len, dtype=torch.bool)
    mask[:, n_prefix:] = torch.rand(2, seq_len - n_prefix) < 0.4
    cls_s, masked_feats = model.forward_student(x, mask=mask)
    print(
        "student CLS:", cls_s.shape, "masked feats:", masked_feats.shape, "(no NaN)",
    )
    assert not torch.isnan(masked_feats).any()

    # Attention helpers
    attn_self = model.teacher_backbone.get_last_selfattention(x)
    attn_cls = model.teacher_backbone.get_cls_attention_map(x, head_fusion="mean")
    print("last self attn:", attn_self.shape)
    print("cls attn map:", attn_cls.shape)

    # Variable input size (mimics local crop in DINOv2 multi-crop training).
    x_small = torch.randn(2, 5, 24, 24)
    cls_small = model(x_small)
    print("small forward (CLS):", cls_small.shape)
    attn_small = model.teacher_backbone.get_cls_attention_map(
        x_small, head_fusion="none"
    )
    print("small cls attn map:", attn_small.shape)

    # Eval mode: augmentations should not be sampled (deterministic).
    model.eval()
    cls_a = model(x)
    cls_b = model(x)
    assert torch.allclose(cls_a, cls_b), "RoPE augment leaked into eval mode."
    print("eval determinism: OK")

    # Train mode with augmentations: sin/cos must change between calls.
    model.train()
    sin1, cos1 = model.teacher_backbone.vit.rope(H=10, W=10)
    sin2, cos2 = model.teacher_backbone.vit.rope(H=10, W=10)
    assert not torch.allclose(sin1, sin2), "RoPE augment did not vary in train mode."
    print("train RoPE augment varies: OK")
