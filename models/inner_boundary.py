from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import _make_norm


class InnerBoundaryGenerator(nn.Module):
    """Generates inner boundary prior from containment relation map with spatial context.

    Uses larger receptive field (dilation=4, kernel=5) to capture nested hierarchical
    structures typical of containment relationships.
    """

    def __init__(self, context_channels: int = 0, hidden: int = 64, dilation: int = 4, kernel_size: int = 5) -> None:
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

    def forward(self, o_containment: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is not None:
            x = torch.cat([o_containment, context], dim=1)
        else:
            x = o_containment
        return self.net(x)
