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
from typing import Any, Dict, List, Optional, Tuple

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


class RoPE2D(nn.Module):
    """Axial 2D Rotary Position Embedding.

    The head dimension ``D`` is split in half: the first ``D/2`` channels are
    rotated using the x-coordinate of the token, the second ``D/2`` using the
    y-coordinate. ``D`` must be divisible by 4 (each axis uses
    LLaMA-style "rotate-half" RoPE which requires the half-axis dim to be
    even).
    """

    def __init__(self, head_dim: int, base: float = 100.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"RoPE2D requires head_dim divisible by 4, got {head_dim}."
            )
        self.head_dim = head_dim
        self.base = float(base)
        axis_dim = head_dim // 2  # dims used per axis
        # Half of axis_dim distinct frequencies, then duplicated to axis_dim
        # via the rotate-half trick.
        freq_dim = axis_dim // 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, freq_dim, dtype=torch.float32) / freq_dim)
        )
        # Persistent so HuggingFace ``from_pretrained`` (which uses meta init
        # and only restores tensors present in the saved state dict) keeps
        # this buffer initialised.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _angle(self, pos: Tensor) -> Tuple[Tensor, Tensor]:
        # pos: (..., 1) scalar position per token along one axis.
        # angles: (..., freq_dim)
        angles = pos * self.inv_freq.to(dtype=pos.dtype, device=pos.device)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        # Duplicate to axis_dim via rotate-half convention.
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        return cos, sin

    def rotate(
        self,
        x: Tensor,
        positions: Tensor,
        prefix_count: int = 0,
    ) -> Tensor:
        """Rotate ``x`` according to 2D positions.

        Args:
            x: Tensor of shape ``(B, h, N, D)``.
            positions: Tensor of shape ``(B, N, 2)`` with ``(x_pos, y_pos)``
                per token. The first ``prefix_count`` tokens are passed
                through unchanged regardless of their stored positions.
            prefix_count: Number of tokens at the start of the sequence that
                should not be rotated.

        Returns:
            Tensor of shape ``(B, h, N, D)`` with the spatial half rotated.
        """
        B, h, N, D = x.shape
        if D != self.head_dim:
            raise ValueError(
                f"RoPE2D expected head_dim={self.head_dim}, got {D}."
            )
        if positions.shape[:2] != (B, N):
            raise ValueError(
                f"positions shape {tuple(positions.shape)} incompatible with "
                f"x shape {tuple(x.shape)}."
            )

        # Split head dim into x-half and y-half.
        Dx = D // 2
        x_x, x_y = x[..., :Dx], x[..., Dx:]

        pos_x = positions[..., 0:1].to(dtype=x.dtype)  # (B, N, 1)
        pos_y = positions[..., 1:2].to(dtype=x.dtype)
        cos_x, sin_x = self._angle(pos_x)  # (B, N, Dx)
        cos_y, sin_y = self._angle(pos_y)  # (B, N, Dx)

        # Broadcast over heads: (B, 1, N, Dx).
        cos_x = cos_x.unsqueeze(1)
        sin_x = sin_x.unsqueeze(1)
        cos_y = cos_y.unsqueeze(1)
        sin_y = sin_y.unsqueeze(1)

        rotated_x = x_x * cos_x + self._rotate_half(x_x) * sin_x
        rotated_y = x_y * cos_y + self._rotate_half(x_y) * sin_y

        if prefix_count > 0:
            # Restore the original (un-rotated) values for prefix tokens.
            rotated_x = torch.cat(
                [x_x[:, :, :prefix_count, :], rotated_x[:, :, prefix_count:, :]],
                dim=2,
            )
            rotated_y = torch.cat(
                [x_y[:, :, :prefix_count, :], rotated_y[:, :, prefix_count:, :]],
                dim=2,
            )

        return torch.cat([rotated_x, rotated_y], dim=-1)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class MultiHeadAttentionRoPE2D(nn.Module):
    """Multi-head attention with 2D RoPE on Q and K (never on V)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: RoPE2D,
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

        # Store ``rope`` as a non-submodule reference. The top-level
        # ``PerceiverStrucPerc`` owns the only ``RoPE2D`` instance; if we
        # registered it here too, every attention block would expose the
        # same ``inv_freq`` buffer at a different state-dict path which
        # confuses HuggingFace's ``save_pretrained``.
        self.__dict__["rope"] = rope

    def forward(
        self,
        q_tokens: Tensor,
        kv_tokens: Tensor,
        q_pos: Tensor,
        kv_pos: Tensor,
        q_prefix_count: int = 0,
        kv_prefix_count: int = 0,
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

        q = self.rope.rotate(q, q_pos, prefix_count=q_prefix_count)
        k = self.rope.rotate(k, kv_pos, prefix_count=kv_prefix_count)

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
        rope: RoPE2D,
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
            rope=rope,
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
        q_pos: Tensor,
        kv_pos: Tensor,
        q_prefix_count: int = 0,
        kv_prefix_count: int = 0,
        return_attn: bool = False,
    ) -> Tensor:
        if return_attn:
            return self.attn(
                self.norm1_q(q),
                self.norm1_kv(kv),
                q_pos,
                kv_pos,
                q_prefix_count=q_prefix_count,
                kv_prefix_count=kv_prefix_count,
                return_attn=True,
            )
        h = self.attn(
            self.norm1_q(q),
            self.norm1_kv(kv),
            q_pos,
            kv_pos,
            q_prefix_count=q_prefix_count,
            kv_prefix_count=kv_prefix_count,
        )
        q = q + self.drop_path1(self.ls1(h))
        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block on the latent tokens (with prefix)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: RoPE2D,
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
            rope=rope,
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
        pos: Tensor,
        prefix_count: int = 0,
        return_attn: bool = False,
    ) -> Tensor:
        x_norm = self.norm1(x)
        if return_attn:
            return self.attn(
                x_norm,
                x_norm,
                pos,
                pos,
                q_prefix_count=prefix_count,
                kv_prefix_count=prefix_count,
                return_attn=True,
            )
        h = self.attn(
            x_norm,
            x_norm,
            pos,
            pos,
            q_prefix_count=prefix_count,
            kv_prefix_count=prefix_count,
        )
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
        rope_base: float = 100.0,
        rope_scale: str = "latent",
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
        self.rope_scale = rope_scale

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

        # 2D RoPE (shared across all attention layers).
        self.rope = RoPE2D(head_dim=self.head_dim, base=rope_base)

        # Drop path schedule (uniform across cross + self blocks initially).
        total_blocks = self.num_cross_blocks * self.depth_outer + self.num_self_blocks
        dp_rates = [drop_path_rate for _ in range(max(total_blocks, 1))]

        # Cross-attention blocks.
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    rope=self.rope,
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
                    rope=self.rope,
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

        # Pre-compute and register positions (in latent-cell units).
        self._build_positions()

        self.apply(_init_weights)

    # Compatibility alias used by code that iterates ``vit.blocks`` (e.g.
    # ``update_drop_path_rate`` and ``get_last_selfattention``). Implemented
    # as a property so it does not create a duplicate submodule path in the
    # state dict (HuggingFace safetensors refuses tied weights).
    @property
    def blocks(self) -> nn.ModuleList:
        return self.self_blocks

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def _build_positions(self) -> None:
        """Pre-compute the latent grid positions (fixed at init).

        Input positions depend on the actual conv output grid (which differs
        between global and local crops), so they are computed on the fly in
        :meth:`_compute_input_pos`. Only the latent grid is cached here.
        """
        latent_h, latent_w = self.latent_size
        if self.rope_scale in ("latent", "input"):
            lat_y = torch.arange(latent_h, dtype=torch.float32) + 0.5
            lat_x = torch.arange(latent_w, dtype=torch.float32) + 0.5
            if self.rope_scale == "input":
                # Latent grid is rescaled to match the configured input grid.
                input_h, input_w = self.patch_embed.input_grid_size
                lat_y = lat_y * (input_h / latent_h)
                lat_x = lat_x * (input_w / latent_w)
        elif self.rope_scale == "normalized":
            lat_y = (torch.arange(latent_h, dtype=torch.float32) + 0.5) / latent_h
            lat_x = (torch.arange(latent_w, dtype=torch.float32) + 0.5) / latent_w
        else:
            raise ValueError(
                f"Unknown rope_scale={self.rope_scale!r}. "
                "Expected one of: 'latent', 'input', 'normalized'."
            )

        yy_lat, xx_lat = torch.meshgrid(lat_y, lat_x, indexing="ij")
        latent_pos = torch.stack([xx_lat, yy_lat], dim=-1).reshape(-1, 2)

        # Persistent so the values survive ``from_pretrained`` round-trips
        # (HF meta-init only fills tensors present in the saved state dict).
        self.register_buffer("latent_pos", latent_pos, persistent=True)

    def _compute_input_pos(
        self, input_h: int, input_w: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Compute input-token positions for an arbitrary (h, w) grid.

        Each crop is treated as a standalone image: the input grid spans
        ``[0, latent_W]`` (or whichever scale ``rope_scale`` selects)
        regardless of the crop's absolute size. This is what we want for the
        DINOv2 multi-crop pipeline where local crops should attend to the
        full latent grid the same way global crops do.
        """
        latent_h, latent_w = self.latent_size

        if self.rope_scale == "latent":
            in_y = (torch.arange(input_h, device=device, dtype=dtype) + 0.5) * (
                latent_h / input_h
            )
            in_x = (torch.arange(input_w, device=device, dtype=dtype) + 0.5) * (
                latent_w / input_w
            )
        elif self.rope_scale == "input":
            in_y = torch.arange(input_h, device=device, dtype=dtype) + 0.5
            in_x = torch.arange(input_w, device=device, dtype=dtype) + 0.5
        elif self.rope_scale == "normalized":
            in_y = (torch.arange(input_h, device=device, dtype=dtype) + 0.5) / input_h
            in_x = (torch.arange(input_w, device=device, dtype=dtype) + 0.5) / input_w
        else:
            raise ValueError(
                f"Unknown rope_scale={self.rope_scale!r}. "
                "Expected one of: 'latent', 'input', 'normalized'."
            )

        yy_in, xx_in = torch.meshgrid(in_y, in_x, indexing="ij")
        return torch.stack([xx_in, yy_in], dim=-1).reshape(-1, 2)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _q_positions(self, batch_size: int) -> Tensor:
        """Positions for prefix + latent queries (prefix entries are zeros)."""
        n_prefix = self.num_prefix_tokens
        prefix_pos = self.latent_pos.new_zeros(n_prefix, 2)
        full = torch.cat([prefix_pos, self.latent_pos], dim=0)  # (n_prefix + L, 2)
        return full.unsqueeze(0).expand(batch_size, -1, -1)

    def _kv_positions(
        self, batch_size: int, input_h: int, input_w: int
    ) -> Tensor:
        pos = self._compute_input_pos(
            input_h, input_w, device=self.latent_pos.device, dtype=self.latent_pos.dtype
        )
        return pos.unsqueeze(0).expand(batch_size, -1, -1)

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
        n_prefix = self.num_prefix_tokens

        in_tokens, (h_in, w_in) = self.patch_embed(images)  # (B, N_in, D)
        q = self.build_initial_queries(B)  # (B, n_prefix + L, D)
        full_q_pos = self._q_positions(B)
        kv_pos = self._kv_positions(B, h_in, w_in)

        intermediates: List[Tensor] = []
        mask_applied = False

        if self.depth_outer == 1:
            for blk in self.cross_blocks:
                q = blk(
                    q,
                    in_tokens,
                    full_q_pos,
                    kv_pos,
                    q_prefix_count=n_prefix,
                    kv_prefix_count=0,
                )
            if mask is not None:
                q = _apply_mask_token(q, mask, mask_token)
                mask_applied = True
            for blk in self.self_blocks:
                q = blk(q, full_q_pos, prefix_count=n_prefix)
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
                        q,
                        in_tokens,
                        full_q_pos,
                        kv_pos,
                        q_prefix_count=n_prefix,
                        kv_prefix_count=0,
                    )
                    cross_idx += 1
                    if mask is not None and not mask_applied:
                        q = _apply_mask_token(q, mask, mask_token)
                        mask_applied = True
                this_outer = self_per_outer + (1 if outer < remainder else 0)
                for _ in range(this_outer):
                    q = self.self_blocks[self_idx](q, full_q_pos, prefix_count=n_prefix)
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
        n_prefix = self.num_prefix_tokens

        if len(self.self_blocks) == 0:
            raise RuntimeError(
                "last_selfattention requires at least one self-attention block."
            )

        in_tokens, (h_in, w_in) = self.patch_embed(images)
        q = self.build_initial_queries(B)
        full_q_pos = self._q_positions(B)
        kv_pos = self._kv_positions(B, h_in, w_in)

        if self.depth_outer == 1:
            for blk in self.cross_blocks:
                q = blk(
                    q,
                    in_tokens,
                    full_q_pos,
                    kv_pos,
                    q_prefix_count=n_prefix,
                    kv_prefix_count=0,
                )
            for blk in self.self_blocks[:-1]:
                q = blk(q, full_q_pos, prefix_count=n_prefix)
            return self.self_blocks[-1](
                q, full_q_pos, prefix_count=n_prefix, return_attn=True
            )

        # Iterated mode: replay the full pipeline up to the last self block.
        self_per_outer, remainder = divmod(self.num_self_blocks, self.depth_outer)
        self_idx = 0
        cross_idx = 0
        last_block = self.self_blocks[-1]
        for outer in range(self.depth_outer):
            for _ in range(self.num_cross_blocks):
                q = self.cross_blocks[cross_idx](
                    q,
                    in_tokens,
                    full_q_pos,
                    kv_pos,
                    q_prefix_count=n_prefix,
                    kv_prefix_count=0,
                )
                cross_idx += 1
            this_outer = self_per_outer + (1 if outer < remainder else 0)
            for k in range(this_outer):
                blk = self.self_blocks[self_idx]
                self_idx += 1
                if blk is last_block:
                    return blk(q, full_q_pos, prefix_count=n_prefix, return_attn=True)
                q = blk(q, full_q_pos, prefix_count=n_prefix)
        raise RuntimeError("Unreachable: failed to reach last self-attention block.")

    @torch.no_grad()
    def first_cross_attention(self, images: Tensor) -> Tensor:
        """Return softmax probs of the first cross-attention block.

        Shape: ``(B, h, n_prefix + L, N_in)``.
        """
        B = images.shape[0]
        n_prefix = self.num_prefix_tokens
        in_tokens, (h_in, w_in) = self.patch_embed(images)
        q = self.build_initial_queries(B)
        full_q_pos = self._q_positions(B)
        kv_pos = self._kv_positions(B, h_in, w_in)
        return self.cross_blocks[0](
            q,
            in_tokens,
            full_q_pos,
            kv_pos,
            q_prefix_count=n_prefix,
            kv_prefix_count=0,
            return_attn=True,
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
        import numpy as np

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
        rope_scale=cfg.get("rope_scale", "latent"),
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
            "rope_scale": "latent",
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
