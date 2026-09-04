from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNormAct, DoubleConv, _make_norm
from .scale_router import ScaleAdaptiveSkip, ScaleRouter
from .seg_head import MultiScaleFeatureSelection


class SemanticAlignment(nn.Module):
    """对齐 skip 特征与当前 decoder 语义空间。"""

    def __init__(self, skip_channels: int, dec_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip_proj = ConvNormAct(skip_channels, out_channels, kernel_size=1)
        self.dec_proj = ConvNormAct(dec_channels, out_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            _make_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, skip_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        # skip_feat: [B, C_skip, H, W]
        # dec_feat:  [B, C_dec,  H', W']
        dec_feat = F.interpolate(dec_feat, size=skip_feat.shape[-2:], mode="bilinear", align_corners=False)
        skip_proj = self.skip_proj(skip_feat)
        dec_proj = self.dec_proj(dec_feat)

        # 利用当前 decoder 语义对 skip 做轻量对齐，缩小 semantic gap。
        semantic_gate = torch.sigmoid((skip_proj * dec_proj).mean(dim=1, keepdim=True))
        aligned = self.refine(torch.cat([skip_proj, dec_proj], dim=1))
        return skip_proj + aligned * semantic_gate


class SelectiveGatedSkip(nn.Module):
    """由 decoder 特征生成门控图，对齐后的 skip 只保留有用细节。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dec_proj = ConvNormAct(channels, channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            _make_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, aligned_skip: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        dec_feat = F.interpolate(dec_feat, size=aligned_skip.shape[-2:], mode="bilinear", align_corners=False)
        dec_proj = self.dec_proj(dec_feat)
        gate_map = self.gate(torch.cat([aligned_skip, dec_proj], dim=1))
        return aligned_skip * gate_map


class FlexiblePriorInjection(nn.Module):
    """Injects only the priors actually provided, avoiding zero-tensor waste."""

    def __init__(self, channels: int, prior_specs: list[tuple[str, int]], shared_boundary_proj: nn.Module | None = None) -> None:
        super().__init__()
        self.dec_proj = ConvNormAct(channels, channels, kernel_size=1)
        self.prior_names = [name for name, _ in prior_specs]
        self.projections = nn.ModuleDict()
        for name, in_ch in prior_specs:
            if name in ('shared', 'inner', 'disjoint') and shared_boundary_proj is not None:
                continue
            self.projections[name] = ConvNormAct(in_ch, channels, kernel_size=1)
        self.shared_boundary_proj = shared_boundary_proj
        self.fuse = nn.Sequential(
            ConvNormAct(channels * (1 + len(prior_specs)), channels, kernel_size=1),
            DoubleConv(channels, channels),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.Sigmoid())
        # Boundary-aware spatial refinement: localizes prior influence to boundary regions
        has_boundary = any(n in ('shared', 'inner', 'disjoint') for n, _ in prior_specs)
        self.boundary_spatial_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        ) if has_boundary else None

    def forward(self, dec_feat: torch.Tensor, priors: dict[str, torch.Tensor | None]) -> torch.Tensor:
        target_size = dec_feat.shape[-2:]
        parts = [self.dec_proj(dec_feat)]
        boundary_feat = None
        for name in self.prior_names:
            prior = priors.get(name)
            if prior is None:
                parts.append(torch.zeros_like(parts[0]))
                continue
            prior = F.interpolate(prior, size=target_size, mode='bilinear', align_corners=False)
            if name in ('shared', 'inner', 'disjoint') and self.shared_boundary_proj is not None:
                parts.append(self.shared_boundary_proj(prior))
                if boundary_feat is None:
                    boundary_feat = prior
                else:
                    boundary_feat = torch.max(boundary_feat, prior)
            else:
                parts.append(self.projections[name](prior))
        fused = self.fuse(torch.cat(parts, dim=1))
        gated = fused * self.gate(fused)
        # Apply boundary-aware spatial refinement
        if self.boundary_spatial_gate is not None and boundary_feat is not None:
            spatial_gate = self.boundary_spatial_gate(boundary_feat)
            gated = gated * (0.5 + 0.5 * spatial_gate)
        return dec_feat + gated


class ImprovedDecoderStage(nn.Module):
    """Single decoder stage: upsample + semantic-aligned skip + gated skip + context + prior injection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        context_channels: int,
        out_channels: int,
        prior_specs: list[tuple[str, int]] | None = None,
        shared_boundary_proj: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_proj = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.skip_align = SemanticAlignment(skip_channels, out_channels, out_channels)
        self.skip_gate = SelectiveGatedSkip(out_channels)
        self.context_proj = ConvNormAct(context_channels, out_channels, kernel_size=1)
        self.fuse = DoubleConv(out_channels * 3, out_channels)
        self.prior_injection = FlexiblePriorInjection(out_channels, prior_specs, shared_boundary_proj) if prior_specs else None

    def forward(
        self,
        x: torch.Tensor,
        skip_feat: torch.Tensor,
        extra_context: Optional[torch.Tensor] = None,
        priors: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip_feat.shape[-2:], mode="bilinear", align_corners=False)
        x = self.input_proj(x)
        aligned_skip = self.skip_align(skip_feat, x)
        gated_skip = self.skip_gate(aligned_skip, x)
        if extra_context is None:
            context_feat = torch.zeros_like(x)
        else:
            extra_context = F.interpolate(extra_context, size=skip_feat.shape[-2:], mode="bilinear", align_corners=False)
            context_feat = self.context_proj(extra_context)
        fused = self.fuse(torch.cat([x, gated_skip, context_feat], dim=1))
        if self.prior_injection is not None and priors is not None:
            fused = self.prior_injection(fused, priors)
        return fused


class ImprovedDecoder(nn.Module):
    """Three-stage decoder with differentiated prior injection per stage.

    Stage3 (H/16→H/8): region + all boundary priors (native resolution match)
    Stage2 (H/8→H/4): edge + shared boundary (edge-related)
    Stage1 (H/4→full): edge only (high-res detail)
    """

    def __init__(self, common_dim: int, decoder_dim: int, use_prior_injection: bool = True, num_classes: int = 1, use_scale_routing: bool = True, dilation_rates: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.shared_boundary_proj = ConvNormAct(1, decoder_dim, kernel_size=1) if use_prior_injection else None

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

        stage3_priors = [('region', common_dim), ('shared', 1)] if use_prior_injection else None
        stage2_priors = [('edge', common_dim), ('shared', 1), ('disjoint', 1)] if use_prior_injection else None
        stage1_priors = [('edge', common_dim)] if use_prior_injection else None

        self.stage3 = ImprovedDecoderStage(
            in_channels=common_dim, skip_channels=common_dim, context_channels=common_dim,
            out_channels=decoder_dim, prior_specs=stage3_priors, shared_boundary_proj=self.shared_boundary_proj,
        )
        self.stage2 = ImprovedDecoderStage(
            in_channels=decoder_dim, skip_channels=common_dim, context_channels=common_dim,
            out_channels=decoder_dim, prior_specs=stage2_priors, shared_boundary_proj=self.shared_boundary_proj,
        )
        self.stage1 = ImprovedDecoderStage(
            in_channels=decoder_dim, skip_channels=common_dim, context_channels=common_dim,
            out_channels=decoder_dim, prior_specs=stage1_priors,
        )

        # Geometry preservation: lightweight residual edge injection at each stage
        self.use_geometry_preserve = use_prior_injection
        if self.use_geometry_preserve:
            self.edge_residual_proj = nn.Conv2d(common_dim, decoder_dim, kernel_size=1, bias=False)
            self.edge_residual_gate = nn.Sequential(
                nn.Conv2d(decoder_dim * 2, 1, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        f_joint: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        f_edge_refined: torch.Tensor | None,
        f_region: torch.Tensor | None,
        b_shared: torch.Tensor | None,
        b_inner: torch.Tensor | None,
        b_disjoint: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (final_output, stage3_output) for deep supervision."""
        if self.scale_router is not None and self.adaptive_skips is not None:
            scale_weights = self.scale_router(f_joint)
            encoder_feats = [f1, f2, f3, f4]
            skip3 = self.adaptive_skips[0](encoder_feats, scale_weights, f3)
            skip2 = self.adaptive_skips[1](encoder_feats, scale_weights, f2)
            skip1 = self.adaptive_skips[2](encoder_feats, scale_weights, f1)
        else:
            skip3, skip2, skip1 = f3, f2, f1

        d3 = self.stage3(f_joint, skip_feat=skip3, extra_context=f4, priors={
            'region': f_region, 'shared': b_shared,
        })
        d2 = self.stage2(d3, skip_feat=skip2, extra_context=f3, priors={
            'edge': f_edge_refined, 'shared': b_shared, 'disjoint': b_disjoint,
        })
        d1 = self.stage1(d2, skip_feat=skip1, extra_context=f2, priors={
            'edge': f_edge_refined,
        })

        # Geometry preservation: residual edge injection before AMFS
        if self.use_geometry_preserve and f_edge_refined is not None:
            edge_res = F.interpolate(f_edge_refined, size=d1.shape[-2:], mode='bilinear', align_corners=False)
            edge_res = self.edge_residual_proj(edge_res)
            geo_gate = self.edge_residual_gate(torch.cat([d1, edge_res], dim=1))
            d1 = d1 + edge_res * geo_gate

        d1 = self.amfs(d1)
        return d1, d3
