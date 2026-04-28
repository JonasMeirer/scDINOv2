from transformers import PretrainedConfig


class ScDINOConfig(PretrainedConfig):
    """Configuration for ScDINO models (DINO and DINOv2 teacher backbones).

    Stores all parameters needed to reconstruct the teacher backbone used for
    inference after self-supervised training.
    """

    model_type = "scdino"

    def __init__(
        self,
        model_variant: str = "dinov2",
        backbone_type: str = "vit",
        # ViT parameters
        in_chans: int = 5,
        img_size: int = 56,
        patch_size: int = 4,
        embed_dim: int = 64,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        reg_tokens: int = 8,
        # ResNet parameters (DINO only)
        stem_width: int = 32,
        # Perceiver / StrucPerc parameters (model_variant="dinov2_StrucPerc")
        latent_h: int = 25,
        latent_w: int = 25,
        num_self_blocks: int = 8,
        num_cross_blocks: int = 1,
        depth_outer: int = 1,
        # 2D RoPE (DINOv3-style)
        rope_base: float | None = 100.0,
        rope_min_period: float | None = None,
        rope_max_period: float | None = None,
        rope_normalize_coords: str = "separate",
        rope_shift_coords: float | None = None,
        rope_jitter_coords: float | None = None,
        rope_rescale_coords: float | None = None,
        init_values: float = 1e-5,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        **kwargs,
    ):
        self.model_variant = model_variant
        self.backbone_type = backbone_type
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.reg_tokens = reg_tokens
        self.stem_width = stem_width
        # Perceiver / StrucPerc fields
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.num_self_blocks = num_self_blocks
        self.num_cross_blocks = num_cross_blocks
        self.depth_outer = depth_outer
        self.rope_base = rope_base
        self.rope_min_period = rope_min_period
        self.rope_max_period = rope_max_period
        self.rope_normalize_coords = rope_normalize_coords
        self.rope_shift_coords = rope_shift_coords
        self.rope_jitter_coords = rope_jitter_coords
        self.rope_rescale_coords = rope_rescale_coords
        self.init_values = init_values
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        super().__init__(**kwargs)
