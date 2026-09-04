from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import _make_norm


class SharedBoundaryGenerator(nn.Module):
    """Generates shared boundary prior from intersection relation map with spatial context.

    Uses medium receptive field (dilation=2, kernel=3) for cross-region shared boundaries.
    """

    def __init__(self, context_channels: int = 0, hidden: int = 48, dilation: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        in_ch = 1 + context_channels
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            _make_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            _make_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, o_intersection: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is not None:
            x = torch.cat([o_intersection, context], dim=1)
        else:
            x = o_intersection
        return self.net(x)
