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
        super().__init__(**kwargs)
