"""Conv encoder that keeps a spatial grid, so a small cube stays localizable."""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128, image_size: int = 224):
        super().__init__()
        channels = [3, 32, 64, 128]
        layers: list[nn.Module] = []
        for inp, out in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.Conv2d(inp, out, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out),
                nn.ReLU(inplace=True),
            ])
        self.net = nn.Sequential(*layers)
        self.reduce = nn.Conv2d(channels[-1], 32, kernel_size=1)
        spatial = image_size // (2 ** (len(channels) - 1))
        self.proj = nn.Linear(32 * spatial * spatial, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.reduce(self.net(image))
        return self.proj(hidden.flatten(1))
