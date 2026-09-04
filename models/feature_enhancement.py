from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNormAct


class HighResImageFusion(nn.Module):
    """Injects original image spatial details into high-resolution encoder features F1/F2."""

    def __init__(self, in_channels: int, common_dim: int) -> None:
        super().__init__()
        self.img_proj_f1 = ConvNormAct(in_channels, common_dim, kernel_size=3)
        self.img_proj_f2 = ConvNormAct(in_channels, common_dim, kernel_size=3)
        self.compress_f1 = ConvNormAct(common_dim * 2, common_dim, kernel_size=1)
        self.compress_f2 = ConvNormAct(common_dim * 2, common_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_f1 = self.img_proj_f1(F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False))
        f1 = self.compress_f1(torch.cat([f1, img_f1], dim=1))

        img_f2 = self.img_proj_f2(F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False))
        f2 = self.compress_f2(torch.cat([f2, img_f2], dim=1))
        return f1, f2


class ParallelDualAttention(nn.Module):
    """Parallel channel + spatial attention using outer-product 2D attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ch_q = nn.Linear(channels, channels)
        self.ch_k = nn.Linear(channels, channels)
        self.ch_v = nn.Conv2d(channels, channels, 1, bias=False)

        self.sp_q = nn.Conv2d(channels, 1, 1)
        self.sp_k = nn.Conv2d(channels, 1, 1)
        self.sp_v = nn.Conv2d(channels, channels, 1, bias=False)

        self.gamma_ch = nn.Parameter(torch.zeros(1))
        self.gamma_sp = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Channel attention: outer product [B,C,1] x [B,1,C] -> [B,C,C]
        gap = x.mean(dim=[-2, -1])  # [B, C]
        q_ch = self.ch_q(gap).unsqueeze(-1)  # [B, C, 1]
        k_ch = self.ch_k(gap).unsqueeze(1)   # [B, 1, C]
        attn_ch = torch.softmax(q_ch @ k_ch / math.sqrt(C), dim=-1)  # [B, C, C]
        v_ch = self.ch_v(x).view(B, C, -1)   # [B, C, HW]
        out_ch = (attn_ch @ v_ch).view(B, C, H, W)

        # Spatial attention: outer product [B,HW,1] x [B,1,HW] -> [B,HW,HW]
        q_sp = self.sp_q(x).view(B, H * W, 1)  # [B, HW, 1]
        k_sp = self.sp_k(x).view(B, 1, H * W)  # [B, 1, HW]
        attn_sp = torch.softmax(q_sp @ k_sp / math.sqrt(H * W), dim=-1)  # [B, HW, HW]
        v_sp = self.sp_v(x).view(B, C, H * W)  # [B, C, HW]
        out_sp = torch.bmm(v_sp, attn_sp.transpose(1, 2)).view(B, C, H, W)

        return x + self.gamma_ch * out_ch + self.gamma_sp * out_sp


class LowResSemanticFusion(nn.Module):
    """F3/F4 concat -> conv -> parallel dual attention -> residual injection back."""

    def __init__(self, common_dim: int) -> None:
        super().__init__()
        self.compress = ConvNormAct(common_dim * 2, common_dim, kernel_size=1)
        self.attention = ParallelDualAttention(common_dim)
        self.proj_f3 = nn.Conv2d(common_dim, common_dim, 1, bias=False)
        self.proj_f4 = nn.Conv2d(common_dim, common_dim, 1, bias=False)
        self.alpha_f3 = nn.Parameter(torch.zeros(1))
        self.alpha_f4 = nn.Parameter(torch.zeros(1))

    def forward(self, f3: torch.Tensor, f4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f4_up = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.compress(torch.cat([f3, f4_up], dim=1))
        enhanced = self.attention(fused)

        f3_new = f3 + self.alpha_f3 * self.proj_f3(enhanced)
        f4_new = f4 + self.alpha_f4 * F.interpolate(self.proj_f4(enhanced), size=f4.shape[-2:], mode="bilinear", align_corners=False)
        return f3_new, f4_new
