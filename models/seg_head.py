from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNormAct, DoubleConv, _make_norm
from .scale_router import ScaleAdaptiveSkip, ScaleRouter


class AdaptivePriorGate(nn.Module):
    """Content-adaptive gating between edge detail and region-derived features.

    Uses edge confidence (activation magnitude) as routing signal to dynamically
    weight edge vs region contributions per spatial location.
    """

    def __init__(self, edge_channels: int, decoder_channels: int) -> None:
        super().__init__()
        self.edge_confidence = nn.Sequential(
            nn.Conv2d(edge_channels, edge_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(edge_channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.edge_proj = nn.Conv2d(edge_channels, decoder_channels, 1, bias=False)
        self.region_proj = nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            _make_norm(decoder_channels),
            nn.GELU(),
        )

    def forward(self, edge_detail: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        edge_detail = F.interpolate(edge_detail, size=decoder_feat.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.edge_confidence(edge_detail)
        e = self.edge_proj(edge_detail)
        r = self.region_proj(decoder_feat)
        fused = alpha * e + (1.0 - alpha) * r
        return self.fuse(fused) + decoder_feat


class MultiScaleFeatureSelection(nn.Module):
    """AMFS-style multi-branch dilated convolution with SE attention selection."""

    def __init__(self, channels: int, dilation_rates: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False),
                _make_norm(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1, bias=False),
            )
            for d in dilation_rates
        ])
        n_branches = len(dilation_rates)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels * n_branches, channels),
            nn.GELU(),
            nn.Linear(channels, channels * n_branches),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * n_branches, channels, 1, bias=False),
            _make_norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        cat = torch.cat(feats, dim=1)
        w = self.se(cat).unsqueeze(-1).unsqueeze(-1)
        return self.proj(cat * w) + x


class GatedSkipConnection(nn.Module):
    """Attention U-Net style gate: decoder context drives skip filtering."""

    def __init__(self, skip_channels: int, gate_channels: int, out_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, out_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        mid = max(out_channels // reduction, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(out_channels, mid),
            nn.GELU(),
            nn.Linear(mid, out_channels),
            nn.Sigmoid(),
        )
        self.norm = _make_norm(out_channels)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        s = self.skip_proj(skip)
        g = self.gate_proj(gate)
        g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        # Spatial attention (Attention U-Net)
        spatial_w = self.psi(F.gelu(s + g))
        s = s * spatial_w
        # Channel attention (SE)
        cw = self.channel_gate(s).unsqueeze(-1).unsqueeze(-1)
        s = s * cw
        return self.norm(s)


class DenseDecoderStage(nn.Module):
    """Decoder stage with context-gated skip + dense connections from prior stages."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, num_prior_stages: int = 0, reduction: int = 4
    ) -> None:
        super().__init__()
        self.gate = GatedSkipConnection(skip_channels, in_channels, out_channels, reduction)
        total_in = in_channels + out_channels + num_prior_stages * out_channels
        self.compress = ConvNormAct(total_in, out_channels, kernel_size=1)
        self.fuse = DoubleConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, prior_outputs: list[torch.Tensor] | None = None) -> torch.Tensor:
        x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gated_skip = self.gate(skip, x)
        parts = [x_up, gated_skip]
        if prior_outputs:
            for p in prior_outputs:
                parts.append(F.interpolate(p, size=skip.shape[-2:], mode="bilinear", align_corners=False))
        return self.fuse(self.compress(torch.cat(parts, dim=1)))


class SegmentationHead(nn.Module):
    """Independent decoder with context-gated skips, dense connections, adaptive prior gate, and AMFS."""

    def __init__(
        self,
        common_dim: int,
        decoder_dim: int,
        num_classes: int,
        task_mode: str = "multilabel",
        dilation_rates: tuple[int, ...] = (1, 3, 5),
        use_scale_routing: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task_mode = task_mode

        self.input_proj = ConvNormAct(common_dim * 2, decoder_dim, kernel_size=1)
        self.high_proj = ConvNormAct(common_dim, common_dim, kernel_size=1)
        self.stage3 = DenseDecoderStage(decoder_dim, common_dim, decoder_dim, num_prior_stages=0)
        self.stage2 = DenseDecoderStage(decoder_dim, common_dim, decoder_dim, num_prior_stages=1)
        self.stage1 = DenseDecoderStage(decoder_dim, common_dim, decoder_dim, num_prior_stages=2)

        self.prior_gate = AdaptivePriorGate(common_dim, decoder_dim)
        self.amfs = MultiScaleFeatureSelection(decoder_dim, dilation_rates=dilation_rates)

        self.scale_router: ScaleRouter | None = None
        self.adaptive_skips: nn.ModuleList | None = None
        if use_scale_routing:
            self.scale_router = ScaleRouter(common_dim, num_scales=4, num_classes=num_classes)
            self.adaptive_skips = nn.ModuleList([
                ScaleAdaptiveSkip(common_dim, num_scales=4),
                ScaleAdaptiveSkip(common_dim, num_scales=4),
                ScaleAdaptiveSkip(common_dim, num_scales=4),
            ])

        self.seg_head = nn.Sequential(
            DoubleConv(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )
        self.aux_head_d3 = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        self.aux_head_d2 = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        edge_detail: torch.Tensor,
        f_high: torch.Tensor,
        input_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        f4_up = F.interpolate(f4, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        x = self.input_proj(torch.cat([bottleneck, f4_up], dim=1))

        f2_enriched = f2 + self.high_proj(f_high)

        if self.scale_router is not None and self.adaptive_skips is not None:
            scale_weights = self.scale_router(bottleneck)
            encoder_feats = [f1, f2_enriched, f3, f4]
            skip3 = self.adaptive_skips[0](encoder_feats, scale_weights, f3)
            skip2 = self.adaptive_skips[1](encoder_feats, scale_weights, f2_enriched)
            skip1 = self.adaptive_skips[2](encoder_feats, scale_weights, f1)
        else:
            skip3, skip2, skip1 = f3, f2_enriched, f1

        d3 = self.stage3(x, skip3, prior_outputs=[])
        d2 = self.stage2(d3, skip2, prior_outputs=[d3])
        d1 = self.stage1(d2, skip1, prior_outputs=[d3, d2])

        d1 = self.prior_gate(edge_detail, d1)
        d1 = self.amfs(d1)

        target_size = input_size if input_size is not None else (f1.shape[-2] * 4, f1.shape[-1] * 4)
        seg_logits = self.seg_head(F.interpolate(d1, size=target_size, mode="bilinear", align_corners=False))
        aux_logits_d3 = self.aux_head_d3(F.interpolate(d3, size=target_size, mode="bilinear", align_corners=False))
        aux_logits_d2 = self.aux_head_d2(F.interpolate(d2, size=target_size, mode="bilinear", align_corners=False))

        return {
            "seg_logits": seg_logits,
            "aux_seg_logits": aux_logits_d3,
            "aux_seg_logits_d2": aux_logits_d2,
        }