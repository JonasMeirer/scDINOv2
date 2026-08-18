import torch
import torch.nn as nn
from typing import Callable

# How the per-channel embeddings are combined. Distinct from
# ``conv_mod.FLAVORS``, which names patch-embedding seeding strategies.
FLAVORS = ("concat", "mean")


class PerChannelWrapper(nn.Module):
    """Run each data channel through the pretrained 3-channel model independently.

    Each single channel is repeated 3× to form a pseudo-RGB image and fed
    through the base model. The resulting embeddings are then combined
    according to ``flavor``:

    - ``"concat"``: concatenated, giving num_channels × base_embedding_dim.
    - ``"mean"``: averaged, giving base_embedding_dim.
    """

    def __init__(
        self, model: nn.Module, extract_fn: Callable, num_channels: int, flavor="concat"
    ):
        super().__init__()
        if flavor not in FLAVORS:
            raise ValueError(f"Unknown flavor {flavor!r}, expected one of {FLAVORS}")
        self.model = model
        self.extract_fn = extract_fn
        self.num_channels = num_channels
        self.flavor = flavor

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        parts = []
        for c in range(self.num_channels):
            rgb = x[:, c : c + 1, :, :].expand(-1, 3, -1, -1)  # (B, 3, H, W)
            emb = self.extract_fn(self.model(rgb))  # (B, D)
            parts.append(emb)
        if self.flavor == "concat":
            return torch.cat(parts, dim=1)  # (B, C*D)
        elif self.flavor == "mean":
            # list to tensor
            parts = torch.stack(parts)  # (C, B, D)
            return torch.mean(parts, dim=0)  # (B, D)
        else:  # unreachable: validated in __init__
            raise ValueError(
                f"Unknown flavor {self.flavor!r}, expected one of {FLAVORS}"
            )
