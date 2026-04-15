import torch
import torch.nn as nn
from typing import Callable


class PerChannelWrapper(nn.Module):
    """Run each data channel through the pretrained 3-channel model independently.

    Each single channel is repeated 3× to form a pseudo-RGB image, fed through
    the base model, and the resulting embeddings are concatenated.
    Output dimension: num_channels × base_embedding_dim.
    """

    def __init__(self, model: nn.Module, extract_fn: Callable, num_channels: int):
        super().__init__()
        self.model = model
        self.extract_fn = extract_fn
        self.num_channels = num_channels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        parts = []
        for c in range(self.num_channels):
            rgb = x[:, c : c + 1, :, :].expand(-1, 3, -1, -1)  # (B, 3, H, W)
            emb = self.extract_fn(self.model(rgb))  # (B, D)
            parts.append(emb)
        return torch.cat(parts, dim=1)  # (B, C*D)
