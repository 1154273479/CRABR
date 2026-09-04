from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNormAct, DoubleConv, WindowAttentionBlock


class PatchEmbedding(nn.Module):
    """Patch embedding that converts the input image to H/4, W/4 tokens."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        hidden_dim = max(embed_dim // 2, 32)
        self.proj = nn.Sequential(
            DoubleConv(in_channels, hidden_dim),
            ConvNormAct(hidden_dim, embed_dim, stride=2),
            ConvNormAct(embed_dim, embed_dim, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PyramidTransformerEncoder(nn.Module):
    """Pyramid transformer encoder.

    Outputs:
        F1: [B, C1, H/4,  W/4]
        F2: [B, C2, H/8,  W/8]
        F3: [B, C3, H/16, W/16]
        F4: [B, C4, H/32, W/32]
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        auto_pad: bool = True,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError('PyramidTransformerEncoder expects 4 stages for depths and num_heads')

        self.auto_pad = auto_pad
        self._pad_info: tuple[int, int] | None = None
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.patch_embed = PatchEmbedding(in_channels, dims[0])
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for stage_idx, (dim, depth, heads) in enumerate(zip(dims, depths, num_heads)):
            blocks = [WindowAttentionBlock(dim, heads, window_size) for _ in range(depth)]
            self.stages.append(nn.Sequential(*blocks))
            if stage_idx < len(dims) - 1:
                self.downsamples.append(ConvNormAct(dim, dims[stage_idx + 1], stride=2))

        self.out_channels = dims

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        H, W = x.shape[-2:]
        self._pad_info = None
        if H % 32 != 0 or W % 32 != 0:
            if self.auto_pad:
                pad_h = (32 - H % 32) % 32
                pad_w = (32 - W % 32) % 32
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
                self._pad_info = (H, W)
            else:
                raise ValueError(
                    f"Input size ({H}, {W}) not divisible by 32. "
                    f"Set auto_pad=True or resize input."
                )
        x = self.patch_embed(x)
        feats: List[torch.Tensor] = []
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)
        if self._pad_info is not None:
            orig_h, orig_w = self._pad_info
            scales = [4, 8, 16, 32]
            feats = [f[..., :orig_h // s, :orig_w // s] for f, s in zip(feats, scales)]
        return feats
