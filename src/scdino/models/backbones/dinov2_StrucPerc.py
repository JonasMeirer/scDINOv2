"""StrucPerc DINOv2 backbone — DINOv3-aligned design.

A Perceiver-style vision encoder for the scDINO codebase.

Differences vs. the standard ViT (``dinov2.py``):

* The input image is patch-embedded onto a (potentially large) input grid,
  e.g. ``input_grid = (50, 50)`` with ``patch_size=1`` on a 50x50 image.
* A learnable bank of latent tokens lives on a smaller, configurable
  spatial grid (e.g. ``latent_size = (25, 25)``).
* Cross-attention with axial 2D RoPE on Q/K (V is **not** rotated) lets
  the latent queries attend to input keys/values. A single sampled
  augmentation state is shared between the latent Q grid and the input
  KV grid so their relative geometry is preserved.
* Latent self-attention blocks (also with 2D RoPE on Q/K) refine the
  latent representation. CLS and storage tokens are concatenated with the
  latent tokens; their RoPE rotation is the identity (no spatial position).
* iBOT masking lives at the **latent** grid: masked latent positions are
  replaced with ``mask_token`` after the first cross-attention block and
  before the latent self-attention stack. This keeps the wrapper a
  drop-in replacement for ``MaskedVisionTransformerTIMM`` w.r.t. the
  existing Lightning training step.

Apart from the Perceiver-specific cross-attention path, this module
delegates as much as possible to DINOv3's components
(:mod:`src.scdino.models.backbones.dinov3`): the inner self-attention
blocks, the FFN classes (Mlp / SwiGLU variants), the norm classes
(LayerNorm / LayerNormBF16 / RMSNorm), the patch embedding, the
``RopePositionEmbedding``, ``LayerScale``, and the weight initialisation
(``init_weights_vit`` via ``named_apply``).

This module exposes:

* ``CrossAttention``, ``CrossAttentionBlock`` (Perceiver-only).
* ``PerceiverPatchEmbed`` (subclass of DINOv3 ``PatchEmbed``; reports the
  **latent** grid as ``grid_size`` so iBOT masking targets latent tokens).
* ``PerceiverStrucPerc`` (the inner backbone, also exposed as the ``vit``
  namespace expected elsewhere in the codebase).
* ``MaskedPerceiverStrucPerc`` (mirrors ``MaskedVisionTransformerTIMM``).
* ``DINOv2StrucPerc`` (top-level model with teacher/student backbones +
  reused projection heads).
"""

from __future__ import annotations

import copy
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from src.scdino.models.backbones.dinov2 import (
    DINOv2Head,
    DINOv2ProjectionHead,
    freeze_eval_module,
)
from src.scdino.models.backbones.dinov3 import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RopePositionEmbedding,
    SelfAttentionBlock,
    dtype_dict,
    ffn_layer_dict,
    init_weights_vit,
    make_2tuple,
    named_apply,
    norm_layer_dict,
    rope_apply,
)


# ---------------------------------------------------------------------------
# Cross-attention (Perceiver-only)
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """Multi-head cross-attention with 2D RoPE on Q and K.

    Mirrors :class:`src.scdino.models.backbones.dinov3.SelfAttention` as
    closely as possible:

    * Same fused-projection style — a single ``kv`` ``Linear(dim, 2*dim)``
      for the input branch, mirroring DINOv3's fused ``qkv``.
    * Same RoPE application via :func:`rope_apply`, with implicit
      prefix-skipping (any leading tokens whose count exceeds the spatial
      sin/cos length are treated as identity-rotation, exactly like
      DINOv3's ``apply_rope``).
    * Same scaled-dot-product attention backend.

    The differences are forced by the cross-attention semantics:
    * Q comes from the latent bank (with prefix CLS / storage tokens),
      K and V come from the input patch tokens (no prefix).
    * ``mask_k_bias`` is **not** supported here — the K-bias-masking trick
      is configured per-block in DINOv3 for the QKV layout. If enabled in
      the config it only applies to the latent **self-attention** blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        device: Any | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias, device=device)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _apply_rope_to(
        x: Tensor, rope: Optional[Tuple[Tensor, Tensor]]
    ) -> Tensor:
        """Rotate ``x[:, :, prefix:, :]`` in-place where
        ``prefix = N - sin.shape[-2]``. Identical bookkeeping to
        ``SelfAttention.apply_rope`` in dinov3.
        """
        if rope is None:
            return x
        sin, cos = rope
        rope_dtype = sin.dtype
        x_dtype = x.dtype
        x = x.to(dtype=rope_dtype)
        N = x.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0, f"sin has more positions ({sin.shape[-2]}) than tokens ({N})"
        x_prefix = x[:, :, :prefix, :]
        x_rot = rope_apply(x[:, :, prefix:, :], sin, cos)
        x = torch.cat((x_prefix, x_rot), dim=-2)
        return x.to(dtype=x_dtype)

    def forward(
        self,
        q_tokens: Tensor,
        kv_tokens: Tensor,
        q_rope: Optional[Tuple[Tensor, Tensor]] = None,
        kv_rope: Optional[Tuple[Tensor, Tensor]] = None,
        return_attn: bool = False,
    ) -> Tensor:
        B, N_q, C = q_tokens.shape
        N_kv = kv_tokens.shape[1]
        h = self.num_heads
        d = C // h

        q = self.q(q_tokens).reshape(B, N_q, h, d).transpose(1, 2)  # (B, h, N_q, d)
        kv = (
            self.kv(kv_tokens)
            .reshape(B, N_kv, 2, h, d)
            .permute(2, 0, 3, 1, 4)
        )  # (2, B, h, N_kv, d)
        k, v = torch.unbind(kv, 0)

        q = self._apply_rope_to(q, q_rope)
        k = self._apply_rope_to(k, kv_rope)

        if return_attn:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            return attn.softmax(dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N_q, C)
        return self.proj_drop(self.proj(x))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block matching DINOv3's
    :class:`SelfAttentionBlock` design:

    * ``LayerScale`` on both residuals (when ``init_values`` is set).
    * Stochastic depth via ``sample_drop_ratio`` (per-batch sample skip
      with ``torch.index_add``), not the timm ``DropPath`` module.
    * Configurable norm layer (``LayerNorm`` / ``LayerNormBF16`` /
      ``RMSNorm``) and FFN layer (``Mlp`` / ``SwiGLUFFN``).
    """

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
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ffn_layer=Mlp,
        device: Any | None = None,
    ) -> None:
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            device=device,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        ffn_hidden = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=ffn_hidden,
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

        self.sample_drop_ratio = float(drop_path)

    def _forward(
        self,
        q: Tensor,
        kv: Tensor,
        q_rope: Optional[Tuple[Tensor, Tensor]],
        kv_rope: Optional[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        b = q.shape[0]
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = torch.randperm(b, device=q.device)[:sample_subset_size]
            q_sub_1 = q[indices_1]
            kv_sub_1 = kv[indices_1]
            residual_1 = self.attn(
                self.norm1_q(q_sub_1),
                self.norm1_kv(kv_sub_1),
                q_rope=q_rope,
                kv_rope=kv_rope,
            )
            q_attn = torch.index_add(
                q,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = torch.randperm(b, device=q.device)[:sample_subset_size]
            q_sub_2 = q_attn[indices_2]
            residual_2 = self.mlp(self.norm2(q_sub_2))
            q_ffn = torch.index_add(
                q_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            q_attn = q + self.ls1(
                self.attn(
                    self.norm1_q(q),
                    self.norm1_kv(kv),
                    q_rope=q_rope,
                    kv_rope=kv_rope,
                )
            )
            q_ffn = q_attn + self.ls2(self.mlp(self.norm2(q_attn)))
        return q_ffn

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        q_rope: Optional[Tuple[Tensor, Tensor]] = None,
        kv_rope: Optional[Tuple[Tensor, Tensor]] = None,
        return_attn: bool = False,
    ) -> Tensor:
        if return_attn:
            return self.attn(
                self.norm1_q(q),
                self.norm1_kv(kv),
                q_rope=q_rope,
                kv_rope=kv_rope,
                return_attn=True,
            )
        return self._forward(q, kv, q_rope, kv_rope)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class PerceiverPatchEmbed(PatchEmbed):
    """Conv2d patch embedding for the Perceiver input branch.

    Subclasses DINOv3's :class:`PatchEmbed` so the same
    ``reset_parameters`` (uniform fan-in init) and ``isinstance``-based
    dispatch in :func:`init_weights_vit` are reused unchanged.

    To keep the existing Lightning iBOT-mask logic (which reads
    ``vit.patch_embed.grid_size`` to size the mask) targeted at the
    **latent** grid, we override ``grid_size`` and ``num_patches`` to
    report the latent grid. The actual conv-output grid is exposed
    through ``input_grid_size`` and ``num_input_patches``.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_size: int,
        img_size: int,
        latent_size: Tuple[int, int],
        device: Any | None = None,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=None,
            flatten_embedding=True,
        )
        # The parent stored its conv-output grid under ``patches_resolution``
        # (and ``grid_size`` via the alias we added in dinov3.py). Stash it
        # under a different name and override grid_size to report the
        # latent grid.
        self.input_grid_size = self.patches_resolution
        self.num_input_patches = (
            self.input_grid_size[0] * self.input_grid_size[1]
        )
        latent_HW = make_2tuple(tuple(latent_size))
        self.grid_size = tuple(latent_HW)
        self.num_patches = latent_HW[0] * latent_HW[1]
        # Move to device if requested (dinov3's PatchEmbed builds on default).
        if device is not None:
            self.to(device)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.proj(x)  # (B, D, H_in, W_in)
        h_in, w_in = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # (B, H_in*W_in, D)
        x = self.norm(x)
        return x, (h_in, w_in)


# ---------------------------------------------------------------------------
# Perceiver
# ---------------------------------------------------------------------------


class PerceiverStrucPerc(nn.Module):
    """Perceiver-style backbone with a spatially structured latent grid.

    Also acts as the ``vit`` namespace exposed by
    :class:`MaskedPerceiverStrucPerc` so that
    ``backbone.vit.patch_embed.grid_size``,
    ``backbone.vit.num_prefix_tokens`` and ``backbone.vit.blocks`` keep
    their TIMM-ViT semantics.
    """

    def __init__(
        self,
        *,
        in_chans: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        latent_size: Tuple[int, int] = (25, 25),
        num_self_blocks: int = 8,
        num_cross_blocks: int = 1,
        depth_outer: int = 1,
        n_storage_tokens: int = 0,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        mask_k_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = 1e-5,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        # 2D RoPE (DINOv3-style)
        rope_base: Optional[float] = 100.0,
        rope_min_period: Optional[float] = None,
        rope_max_period: Optional[float] = None,
        rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        rope_shift_coords: Optional[float] = None,
        rope_jitter_coords: Optional[float] = None,
        rope_rescale_coords: Optional[float] = None,
        rope_dtype: str = "fp32",
        device: Any | None = None,
    ) -> None:
        super().__init__()
        if depth_outer < 1:
            raise ValueError("depth_outer must be >= 1.")
        if num_cross_blocks < 1:
            raise ValueError("num_cross_blocks must be >= 1.")
        if num_self_blocks < 0:
            raise ValueError("num_self_blocks must be >= 0.")

        self.embed_dim = embed_dim
        self.num_features = embed_dim  # for parity with DinoVisionTransformer
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_size = make_2tuple(tuple(latent_size))
        self.latent_h, self.latent_w = self.latent_size
        self.L = self.latent_h * self.latent_w
        self.n_storage_tokens = n_storage_tokens

        self.depth_outer = int(depth_outer)
        self.num_cross_blocks = int(num_cross_blocks)
        self.num_self_blocks = int(num_self_blocks)
        self.n_blocks = self.num_self_blocks  # parity attribute
        self.patch_size = patch_size

        # Compatibility shims for callers that reach for TIMM-ViT
        # attributes on ``vit``.
        self.has_class_token = True
        self.no_embed_class = True
        self.global_pool = ""
        self.attn_pool = None
        self.dynamic_img_size = False
        # ``pos_embed`` is queried by some lightly utilities; we never use
        # it (RoPE is applied inside attention) but keep the attribute.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, embed_dim, device=device), requires_grad=False
        )

        # Patch embed (input branch).
        self.patch_embed = PerceiverPatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            latent_size=self.latent_size,
            device=device,
        )

        # Learnable latent token bank.
        self.latent_tokens = nn.Parameter(
            torch.empty(1, self.L, embed_dim, device=device)
        )

        # CLS + storage (registers).
        self.cls_token = nn.Parameter(
            torch.empty(1, 1, embed_dim, device=device)
        )
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, self.n_storage_tokens, embed_dim, device=device)
            )
        else:
            self.storage_tokens = None

        # iBOT mask token (lives at the latent grid).
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        # 2D RoPE (shared across all attention layers; only the parent
        # Perceiver owns the ``periods`` buffer).
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=rope_base,
            min_period=rope_min_period,
            max_period=rope_max_period,
            normalize_coords=rope_normalize_coords,
            shift_coords=rope_shift_coords,
            jitter_coords=rope_jitter_coords,
            rescale_coords=rope_rescale_coords,
            dtype=dtype_dict[rope_dtype],
            device=device,
        )

        # Norm / FFN / activation choices (same dispatch tables as dinov3).
        norm_layer_cls = norm_layer_dict[norm_layer]
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        act_layer = nn.GELU

        # Cross-attention blocks (Perceiver-only).
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    init_values=layerscale_init,
                    act_layer=act_layer,
                    norm_layer=norm_layer_cls,
                    ffn_layer=ffn_layer_cls,
                    device=device,
                )
                for _ in range(self.num_cross_blocks * self.depth_outer)
            ]
        )

        # Self-attention blocks (latent + prefix). Reuses DINOv3's block
        # directly so the inner attention, dropout/drop-path, LayerScale,
        # FFN, and ``init_weights_vit`` dispatch all match DINOv3 exactly.
        self.self_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    init_values=layerscale_init,
                    act_layer=act_layer,
                    norm_layer=norm_layer_cls,
                    ffn_layer=ffn_layer_cls,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for _ in range(self.num_self_blocks)
            ]
        )

        # Final norm and (no-op) pre-norm/dropout for compat.
        self.norm = norm_layer_cls(embed_dim)
        self.norm_pre = nn.Identity()
        self.pos_drop = nn.Identity()
        self.head = nn.Identity()  # parity with DinoVisionTransformer

        self.init_weights()

    @property
    def num_prefix_tokens(self) -> int:
        return 1 + self.n_storage_tokens

    @property
    def reg_token(self) -> Optional[Tensor]:
        """Alias for ``storage_tokens`` (TIMM-style)."""
        return self.storage_tokens if self.n_storage_tokens > 0 else None

    # Compatibility alias used by code that iterates ``vit.blocks``
    # (e.g. ``get_last_selfattention``). Implemented as a property so it
    # does not duplicate submodules in the state dict (HuggingFace
    # safetensors refuses tied weights).
    @property
    def blocks(self) -> nn.ModuleList:
        return self.self_blocks

    # ------------------------------------------------------------------
    # Init weights (matches DINOv3)
    # ------------------------------------------------------------------

    def init_weights(self) -> None:
        """Initialize weights using the DINOv3 recipe.

        * ``periods`` of the RoPE module are reset (closed-form).
        * ``cls_token``, ``storage_tokens``, ``latent_tokens`` are
          ``nn.init.normal_(std=0.02)`` (matching DINOv3's
          ``cls_token`` / ``storage_tokens``).
        * ``mask_token`` is zeroed.
        * Every other layer is dispatched via :func:`init_weights_vit`
          (trunc-normal Linear weights, zero biases, default LayerNorm,
          LayerScale init to ``init_values``, PatchEmbed uniform init).
        """
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.normal_(self.latent_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------

    def _compute_q_kv_rope(
        self, h_in: int, w_in: int
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Compute matched RoPE ``(sin, cos)`` for the latent and input
        grids using a **single** sampled augmentation state. Sharing the
        state across the two grids is required for the latent queries
        and input keys to live in a consistent coordinate system; if the
        two ``forward`` calls sampled independently, training-time
        ``shift`` / ``jitter`` / ``rescale`` would create independent
        perturbations and corrupt the relative geometry.
        """
        augment_state = self.rope_embed.sample_augment_state()
        q_rope = self.rope_embed(
            H=self.latent_h, W=self.latent_w, augment_state=augment_state
        )
        kv_rope = self.rope_embed(H=h_in, W=w_in, augment_state=augment_state)
        return q_rope, kv_rope

    def build_initial_queries(self, batch_size: int) -> Tensor:
        """Construct the initial (prefix + latent) query sequence.

        Order: ``[cls, storage_tokens..., latent_tokens]``.
        """
        latent = self.latent_tokens.expand(batch_size, -1, -1)
        prefix = [self.cls_token.expand(batch_size, -1, -1)]
        if self.storage_tokens is not None:
            prefix.append(self.storage_tokens.expand(batch_size, -1, -1))
        prefix_t = torch.cat(prefix, dim=1)
        return torch.cat([prefix_t, latent], dim=1)

    # ------------------------------------------------------------------
    # Forward / encode
    # ------------------------------------------------------------------

    def encode(
        self,
        images: Tensor,
        mask: Optional[Tensor] = None,
        return_intermediates: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """Run the Perceiver pipeline and return the final token sequence.

        Args:
            images: ``(B, C, H, W)``.
            mask: Optional ``(B, n_prefix + L)`` boolean mask. Tokens
                where ``mask`` is True are replaced by ``mask_token``
                after the first cross-attention block.
            return_intermediates: If True, returns a list of all
                intermediate self-attention outputs (after each block,
                pre-norm).

        Returns:
            ``(B, n_prefix + L, D)`` final tokens (after final ``norm``).
            If ``return_intermediates`` is True, also returns the list
            of per-self-block outputs.
        """
        B = images.shape[0]

        in_tokens, (h_in, w_in) = self.patch_embed(images)  # (B, N_in, D)
        q = self.build_initial_queries(B)  # (B, n_prefix + L, D)
        q_rope, kv_rope = self._compute_q_kv_rope(h_in, w_in)

        intermediates: List[Tensor] = []
        mask_applied = False

        if self.depth_outer == 1:
            for blk in self.cross_blocks:
                q = blk(q, in_tokens, q_rope=q_rope, kv_rope=kv_rope)
            if mask is not None:
                q = _apply_mask_token(q, mask, self.mask_token)
                mask_applied = True
            for blk in self.self_blocks:
                q = blk(q, q_rope)
                if return_intermediates:
                    intermediates.append(q)
        else:
            # Iterated mode: depth_outer x (cross blocks then a slice of
            # self blocks). Mask is applied once after the very first
            # cross block.
            self_per_outer, remainder = divmod(
                self.num_self_blocks, self.depth_outer
            )
            self_idx = 0
            cross_idx = 0
            for outer in range(self.depth_outer):
                for _ in range(self.num_cross_blocks):
                    q = self.cross_blocks[cross_idx](
                        q, in_tokens, q_rope=q_rope, kv_rope=kv_rope
                    )
                    cross_idx += 1
                    if mask is not None and not mask_applied:
                        q = _apply_mask_token(q, mask, self.mask_token)
                        mask_applied = True
                this_outer = self_per_outer + (1 if outer < remainder else 0)
                for _ in range(this_outer):
                    q = self.self_blocks[self_idx](q, q_rope)
                    if return_intermediates:
                        intermediates.append(q)
                    self_idx += 1

        q = self.norm(q)

        if return_intermediates:
            return q, intermediates
        return q

    # ------------------------------------------------------------------
    # Attention helpers (visualization)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def last_selfattention(self, images: Tensor) -> Tensor:
        """Return softmax attention probs of the last self-attention block.

        Shape: ``(B, num_heads, n_prefix + L, n_prefix + L)``.

        DINOv3's :class:`SelfAttention` uses fused SDPA which never
        materializes the attention matrix, so we manually recompute Q/K
        for the last block (with RoPE) and softmax to recover the weights.
        """
        if len(self.self_blocks) == 0:
            raise RuntimeError(
                "last_selfattention requires at least one self-attention block."
            )

        B = images.shape[0]
        in_tokens, (h_in, w_in) = self.patch_embed(images)
        q = self.build_initial_queries(B)
        q_rope, kv_rope = self._compute_q_kv_rope(h_in, w_in)

        if self.depth_outer == 1:
            for blk in self.cross_blocks:
                q = blk(q, in_tokens, q_rope=q_rope, kv_rope=kv_rope)
            for blk in self.self_blocks[:-1]:
                q = blk(q, q_rope)
            return _last_selfattention_softmax(self.self_blocks[-1], q, q_rope)

        # Iterated mode: replay the full pipeline up to the last self block.
        self_per_outer, remainder = divmod(self.num_self_blocks, self.depth_outer)
        self_idx = 0
        cross_idx = 0
        last_block = self.self_blocks[-1]
        for outer in range(self.depth_outer):
            for _ in range(self.num_cross_blocks):
                q = self.cross_blocks[cross_idx](
                    q, in_tokens, q_rope=q_rope, kv_rope=kv_rope
                )
                cross_idx += 1
            this_outer = self_per_outer + (1 if outer < remainder else 0)
            for _ in range(this_outer):
                blk = self.self_blocks[self_idx]
                self_idx += 1
                if blk is last_block:
                    return _last_selfattention_softmax(blk, q, q_rope)
                q = blk(q, q_rope)
        raise RuntimeError("Unreachable: failed to reach last self-attention block.")

    @torch.no_grad()
    def first_cross_attention(self, images: Tensor) -> Tensor:
        """Return softmax probs of the first cross-attention block.

        Shape: ``(B, num_heads, n_prefix + L, N_in)``.
        """
        B = images.shape[0]
        in_tokens, (h_in, w_in) = self.patch_embed(images)
        q = self.build_initial_queries(B)
        q_rope, kv_rope = self._compute_q_kv_rope(h_in, w_in)
        return self.cross_blocks[0](
            q, in_tokens, q_rope=q_rope, kv_rope=kv_rope, return_attn=True
        )


def _apply_mask_token(tokens: Tensor, mask: Tensor, mask_token: Tensor) -> Tensor:
    """Replace tokens at True positions of ``mask`` with ``mask_token``.

    ``mask_token`` is allowed to be either ``(D,)``, ``(1, D)`` or
    ``(1, 1, D)``: it is broadcast against ``tokens`` of shape
    ``(B, N, D)`` via standard broadcasting rules after a single
    ``mask.unsqueeze(-1)``.
    """
    # Make sure mask_token has a broadcastable layout: end up with shape
    # (1, 1, D) regardless of how it was created (DINOv3 stores it as
    # (1, D), older code as (1, 1, D)).
    while mask_token.dim() < 3:
        mask_token = mask_token.unsqueeze(0)
    m = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    return tokens * (1.0 - m) + m * mask_token.to(dtype=tokens.dtype)


@torch.no_grad()
def _last_selfattention_softmax(
    block: SelfAttentionBlock,
    x: Tensor,
    rope: Optional[Tuple[Tensor, Tensor]],
) -> Tensor:
    """Recompute the softmax attention weights of one DINOv3
    :class:`SelfAttentionBlock` block. Mirrors the equivalent helper in
    :class:`MaskedDinoVisionTransformer.get_last_selfattention`.
    """
    x_norm = block.norm1(x)
    a = block.attn
    B, N, _ = x_norm.shape
    C = a.qkv.in_features
    qkv = a.qkv(x_norm).reshape(B, N, 3, a.num_heads, C // a.num_heads)
    q, k, _ = torch.unbind(qkv, 2)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    if rope is not None:
        q, k = a.apply_rope(q, k, rope)
    attn = (q @ k.transpose(-2, -1)) * a.scale
    return attn.softmax(dim=-1)


# ---------------------------------------------------------------------------
# Wrapper (mirrors MaskedVisionTransformerTIMM)
# ---------------------------------------------------------------------------


class MaskedPerceiverStrucPerc(nn.Module):
    """Drop-in replacement for ``MaskedVisionTransformerTIMM``.

    Provides ``encode``, ``forward``, ``forward_intermediates``,
    ``get_last_selfattention``, ``get_cls_attention_map``, the
    ``sequence_length`` property and the ``vit`` namespace expected by
    the rest of the codebase.
    """

    def __init__(
        self,
        perceiver: PerceiverStrucPerc,
        mask_token: Optional[Parameter] = None,
    ) -> None:
        super().__init__()
        self.perceiver = perceiver
        # Same trick as MaskedDinoVisionTransformer: if the caller passes
        # a custom ``mask_token``, replace the one inside the perceiver
        # so the parameter still lives at exactly one state-dict path.
        if mask_token is not None:
            self.perceiver.mask_token = mask_token

    # Property to avoid duplicating submodules in the state dict.
    @property
    def vit(self) -> PerceiverStrucPerc:
        return self.perceiver

    @property
    def mask_token(self) -> Parameter:
        return self.perceiver.mask_token

    @property
    def sequence_length(self) -> int:
        return self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens

    def images_to_tokens(self, images: Tensor) -> Tensor:
        """Return the input branch's patch tokens (not used in encode)."""
        tokens, _ = self.vit.patch_embed(images)
        return tokens

    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        prefix = [self.vit.cls_token.expand(x.shape[0], -1, -1)]
        if self.vit.storage_tokens is not None:
            prefix.append(self.vit.storage_tokens.expand(x.shape[0], -1, -1))
        return torch.cat(prefix + [x], dim=1)

    def add_pos_embed(self, x: Tensor) -> Tensor:
        # No additive position embedding: 2D RoPE is applied inside attention.
        return x

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
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
        return self.perceiver.encode(images, mask=mask)

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
        return self.perceiver.last_selfattention(images)

    @torch.no_grad()
    def get_cls_attention_map(
        self, images: Tensor, head_fusion: str = "mean"
    ) -> Tensor:
        """Return the CLS query's cross-attention over input tokens.

        For a Perceiver this is the natural CLS->image heatmap. The
        result is reshaped to the **input** patch grid.

        Shape:
            * ``"mean"`` -> ``(B, 1, H_in, W_in)``
            * ``"max"`` -> ``(B, 1, H_in, W_in)``
            * ``"none"`` -> ``(B, num_heads, H_in, W_in)``
        """
        attn = self.perceiver.first_cross_attention(images)  # (B, h, P+L, N_in)
        cls_to_input = attn[:, :, 0, :]
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
    """Update ``sample_drop_ratio`` on every cross + self attention block.

    Matches :func:`update_drop_path_rate_dinov3` in style (no DropPath
    module — DINOv3 implements stochastic depth via per-batch sample skip
    and ``torch.index_add``).
    """
    import numpy as np

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
        blk.sample_drop_ratio = float(p)


# ---------------------------------------------------------------------------
# Top-level model (mirrors DINOv2)
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
        n_storage_tokens=cfg.get("n_storage_tokens", cfg.get("reg_tokens", 0)),
        ffn_ratio=cfg.get("ffn_ratio", cfg.get("mlp_ratio", 4.0)),
        qkv_bias=cfg.get("qkv_bias", True),
        proj_bias=cfg.get("proj_bias", True),
        ffn_bias=cfg.get("ffn_bias", True),
        mask_k_bias=cfg.get("mask_k_bias", False),
        drop_rate=cfg.get("drop_rate", 0.0),
        attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=0.0,  # student gets drop path applied separately
        layerscale_init=cfg.get(
            "layerscale_init", cfg.get("init_values", 1e-5)
        ),
        norm_layer=cfg.get("norm_layer", "layernorm"),
        ffn_layer=cfg.get("ffn_layer", "mlp"),
        rope_base=cfg.get("rope_base", 100.0),
        rope_min_period=cfg.get("rope_min_period", None),
        rope_max_period=cfg.get("rope_max_period", None),
        rope_normalize_coords=cfg.get("rope_normalize_coords", "separate"),
        rope_shift_coords=cfg.get("rope_shift_coords", None),
        rope_jitter_coords=cfg.get("rope_jitter_coords", None),
        rope_rescale_coords=cfg.get("rope_rescale_coords", None),
        rope_dtype=cfg.get("rope_dtype", "fp32"),
    )


class DINOv2StrucPerc(nn.Module):
    """Top-level DINOv2 model with a StrucPerc backbone.

    Mirrors :class:`src.scdino.models.backbones.dinov2.DINOv2`: builds a
    teacher / student pair of backbones and projection heads, freezes the
    teacher, and applies the drop-path schedule to the student.
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
            "n_storage_tokens": 4,
            "ffn_ratio": 4.0,
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

    model = DINOv2StrucPerc(backbone_config, head_cfg, head_cfg)
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
    sin1, cos1 = model.teacher_backbone.vit.rope_embed(H=10, W=10)
    sin2, cos2 = model.teacher_backbone.vit.rope_embed(H=10, W=10)
    assert not torch.allclose(sin1, sin2), (
        "RoPE augment did not vary in train mode."
    )
    print("train RoPE augment varies: OK")

    # SwiGLU + RMSNorm path (DINOv3-style alt configuration).
    backbone_config_alt = copy.deepcopy(backbone_config)
    backbone_config_alt["perceiver"]["ffn_layer"] = "swiglu"
    backbone_config_alt["perceiver"]["norm_layer"] = "rmsnorm"
    backbone_config_alt["perceiver"]["mask_k_bias"] = True
    alt = DINOv2StrucPerc(backbone_config_alt, head_cfg, head_cfg)
    alt.eval()
    out_alt = alt(x)
    print("swiglu+rmsnorm+mask_k_bias forward (CLS):", out_alt.shape)
