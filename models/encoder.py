from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _make_norm(channels: int) -> nn.Module:
    """GroupNorm with 8 groups — works with any batch size including 1."""
    num_groups = min(8, channels)
    while channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            _make_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvNormAct(in_channels, out_channels), ConvNormAct(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"WindowAttentionBlock requires dim ({dim}) to be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def _window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws).permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.view(-1, ws * ws, c)
        return windows, (h, w, hp, wp)

    def _window_reverse(self, windows: torch.Tensor, shape: Tuple[int, int, int, int], batch_size: int) -> torch.Tensor:
        h, w, hp, wp = shape
        ws = self.window_size
        x = windows.view(batch_size, hp // ws, wp // ws, ws, ws, self.dim)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, self.dim, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        windows, shape = self._window_partition(x)
        residual = windows
        windows = self.norm1(windows)
        qkv = self.qkv(windows).reshape(windows.shape[0], windows.shape[1], 3, self.num_heads, self.dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(windows.shape[0], windows.shape[1], self.dim)
        windows = residual + self.proj(out)
        windows = windows + self.ffn(self.norm2(windows))
        return self._window_reverse(windows, shape, b)


class ResNet50Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, use_pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        backbone = resnet50(weights=weights)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if old_conv.weight.shape[1] == 3 and in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            backbone.conv1 = new_conv
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stem(x)
        x = self.pool(f1)
        f2 = self.layer1(x)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return [f1, f2, f3, f4, f5]


class SwinTinyEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_dim: int = 96, depth: int = 2, num_heads: Sequence[int] = (3, 6, 12, 16)) -> None:
        super().__init__()
        self.stem = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(ConvNormAct(64, base_dim, stride=2), *[WindowAttentionBlock(base_dim, num_heads[0], 7) for _ in range(depth)])
        self.down2 = nn.Sequential(ConvNormAct(base_dim, base_dim * 2, stride=2), *[WindowAttentionBlock(base_dim * 2, num_heads[1], 7) for _ in range(depth)])
        self.down3 = nn.Sequential(ConvNormAct(base_dim * 2, base_dim * 4, stride=2), *[WindowAttentionBlock(base_dim * 4, num_heads[2], 7) for _ in range(depth)])
        self.down4 = nn.Sequential(ConvNormAct(base_dim * 4, 512, stride=2), *[WindowAttentionBlock(512, num_heads[3], 7) for _ in range(depth)])
        self.out_channels = [64, base_dim, base_dim * 2, base_dim * 4, 512]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stem(F.avg_pool2d(x, kernel_size=2, stride=2))
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        return [f1, f2, f3, f4, f5]
