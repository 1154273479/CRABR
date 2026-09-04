from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNormAct, DoubleConv, WindowAttentionBlock, _make_norm


class TemperatureSigmoid(nn.Module):
    """Sigmoid with learnable temperature for softer gating."""

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x / self.temperature)


class HierarchicalCrossLevelFusion(nn.Module):
    """Hierarchical fusion: F1+F2 at H/8, F3+F4 at H/16, then merge."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse_high = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1),
            ConvNormAct(channels, channels),
        )
        self.fuse_low = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1),
            ConvNormAct(channels, channels),
        )
        self.gate_high_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.gate_high_act = TemperatureSigmoid(1.5)
        self.gate_low_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.gate_low_act = TemperatureSigmoid(1.5)
        self.final_fuse = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1),
            DoubleConv(channels, channels),
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        f1, f2, f3, f4 = feats
        # High-res pair: F1+F2 at H/8
        h8_size = f2.shape[-2:]
        f1_h8 = F.interpolate(f1, size=h8_size, mode='bilinear', align_corners=False)
        high = self.fuse_high(torch.cat([f1_h8, f2], dim=1))
        g_high = self.gate_high_act(self.gate_high_conv(high))
        high = high * g_high

        # Low-res pair: F3+F4 at H/16
        h16_size = f3.shape[-2:]
        f4_h16 = F.interpolate(f4, size=h16_size, mode='bilinear', align_corners=False)
        low = self.fuse_low(torch.cat([f3, f4_h16], dim=1))
        g_low = self.gate_low_act(self.gate_low_conv(low))
        low = low * g_low
        self._last_gate_activations = [g_high, g_low]

        # Merge: downsample high to H/16 and fuse with low
        high_down = F.interpolate(high, size=h16_size, mode='bilinear', align_corners=False)
        f_fuse = self.final_fuse(torch.cat([high_down, low], dim=1))
        return f_fuse, high  # f_fuse at H/16, high at H/8


class GatedCrossLevelFusion(nn.Module):
    """Aligns F1-F4 to F3 scale and produces shared context F_fuse."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate_convs = nn.ModuleList([nn.Conv2d(channels, 1, kernel_size=1) for _ in range(4)])
        self.gate_acts = nn.ModuleList([TemperatureSigmoid(1.5) for _ in range(4)])
        self.fuse = nn.Sequential(
            ConvNormAct(channels * 4, channels, kernel_size=1),
            DoubleConv(channels, channels),
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats) != 4:
            raise ValueError(f'GatedCrossLevelFusion expects 4 features, got {len(feats)}')

        target_size = feats[2].shape[-2:]  # F3 scale: [H/16, W/16]
        aligned = []
        gate_activations = []
        for feat, conv, act in zip(feats, self.gate_convs, self.gate_acts):
            resized = feat if feat.shape[-2:] == target_size else F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            g = act(conv(resized))
            gate_activations.append(g)
            aligned.append(resized * g)
        self._last_gate_activations = gate_activations
        return self.fuse(torch.cat(aligned, dim=1))


class SoftEdgeBranch(nn.Module):
    """Lightweight edge branch using high-resolution features F1/F2 and shared F_fuse."""

    def __init__(self, channels: int, edge_classes: int, task_mode: str, tau: float = 2.0) -> None:
        super().__init__()
        self.tau = tau
        self.edge_classes = edge_classes
        self.task_mode = task_mode
        hidden_dim = max(channels // 2, 32)
        self.feature_blocks = nn.ModuleList([ConvNormAct(channels, hidden_dim) for _ in range(3)])
        self.logit_heads = nn.ModuleList([nn.Conv2d(hidden_dim, edge_classes, kernel_size=1) for _ in range(3)])
        self.fuse = nn.Sequential(
            ConvNormAct(hidden_dim * 3, channels, kernel_size=1),
            DoubleConv(channels, channels),
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # feats: [F1, F2, F_fuse]
        target_size = feats[0].shape[-2:]  # Edge features stay at H/4 scale.
        edge_features = []
        edge_logits = []
        for feat, block, head in zip(feats, self.feature_blocks, self.logit_heads):
            edge_feat = block(feat)
            edge_features.append(F.interpolate(edge_feat, size=target_size, mode='bilinear', align_corners=False))
            edge_logits.append(head(edge_feat))

        f_edge = self.fuse(torch.cat(edge_features, dim=1))
        final_logits = edge_logits[-1]  # [B, C_edge, H/16, W/16]
        if self.task_mode in {'exclusive', 'independent', 'binary', 'multilabel'} and self.edge_classes == 1:
            a_edge = torch.sigmoid(final_logits)
        elif self.task_mode in {'independent', 'multilabel'}:
            a_edge = torch.sigmoid(final_logits)
        else:
            a_edge = torch.softmax(final_logits / self.tau, dim=1)
        return f_edge, a_edge, edge_logits


class RegionBranch(nn.Module):
    """Unified region branch for single-label and multi-label region modeling."""

    def __init__(self, channels: int, region_num_classes: int, region_mode: str) -> None:
        super().__init__()
        self.region_num_classes = region_num_classes
        self.region_mode = region_mode
        hidden_dim = max(channels // 2, 32)
        self.feature_blocks = nn.ModuleList([ConvNormAct(channels, hidden_dim) for _ in range(3)])
        self.logit_heads = nn.ModuleList([nn.Conv2d(hidden_dim, region_num_classes, kernel_size=1) for _ in range(3)])
        self.fuse = nn.Sequential(
            ConvNormAct(hidden_dim * 3, channels, kernel_size=1),
            DoubleConv(channels, channels),
        )

        if region_mode == 'single_label':
            self.final_act: nn.Module = nn.Softmax(dim=1)
        elif region_mode == 'multi_label':
            self.final_act = nn.Sigmoid()
        else:
            raise ValueError(f'Unsupported region_mode: {region_mode}')

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # feats: [F3, F4, F_fuse]
        target_size = feats[0].shape[-2:]  # Region features stay at H/16 scale.
        region_features = []
        region_logits = []
        for feat, block, head in zip(feats, self.feature_blocks, self.logit_heads):
            region_feat = block(feat)
            region_features.append(F.interpolate(region_feat, size=target_size, mode='bilinear', align_corners=False))
            region_logits.append(head(region_feat))

        f_region = self.fuse(torch.cat(region_features, dim=1))
        region_logits_out = edge_aligned_logits = region_logits[-1]
        if edge_aligned_logits.shape[-2:] != target_size:
            region_logits_out = F.interpolate(edge_aligned_logits, size=target_size, mode='bilinear', align_corners=False)
        a_region = self.final_act(region_logits_out)
        return f_region, a_region, region_logits_out


class AdaptivePriorAttention(nn.Module):
    """Adaptive attention over edge, region and relation-prior streams.

    The module predicts one spatially varying weight map per prior stream and
    rescales projected features before the ERGA concat fusion. We multiply by
    num_priors * softmax(weights) so the average feature magnitude stays close
    to the original concat path.
    """

    def __init__(self, hidden_dim: int, num_priors: int = 6, reduction: int = 4) -> None:
        super().__init__()
        self.num_priors = num_priors
        mid_dim = max(hidden_dim // reduction, 16)
        self.global_score = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )
        self.spatial_score = nn.Sequential(
            nn.Conv2d(num_priors, num_priors, kernel_size=3, padding=1, bias=False),
            _make_norm(num_priors),
            nn.GELU(),
            nn.Conv2d(num_priors, num_priors, kernel_size=1),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, priors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(priors) != self.num_priors:
            raise ValueError(f'AdaptivePriorAttention expects {self.num_priors} priors, got {len(priors)}')
        stacked = torch.stack(list(priors), dim=1)  # [B, P, C, H, W]
        descriptors = stacked.mean(dim=(-2, -1))  # [B, P, C]
        global_logits = self.global_score(descriptors).squeeze(-1)  # [B, P]
        spatial_logits = self.spatial_score(stacked.mean(dim=2))  # [B, P, H, W]
        logits = global_logits[:, :, None, None] + spatial_logits
        weights = torch.softmax(logits, dim=1)
        self.last_weights = weights.detach()
        scales = weights * float(self.num_priors)
        return [prior * scales[:, idx:idx + 1] for idx, prior in enumerate(priors)]


class ERGAModule(nn.Module):
    """ERGA fusion block with dual-scale paths for edge and region priors."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        use_prior_attention: bool = False,
        relation_types: tuple[str, ...] = ('disjoint', 'intersection', 'containment'),
    ) -> None:
        super().__init__()
        self.use_prior_attention = use_prior_attention
        self.relation_types = list(relation_types)
        num_boundaries = len(relation_types)
        self.edge_proj = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.region_proj = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.boundary_base_proj = nn.Conv2d(1, hidden_dim, kernel_size=1, bias=False)
        self.boundary_adapters = nn.ModuleDict({
            name: nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            for name in relation_types
        })
        self.fuse_proj = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        num_priors = 3 + num_boundaries  # edge + region + fuse + N boundaries
        self.prior_attention = AdaptivePriorAttention(hidden_dim, num_priors=num_priors) if use_prior_attention else None
        groups = max(1, channels // 32)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * num_priors, channels, kernel_size=1, bias=False),
            _make_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=False),
            _make_norm(channels),
            nn.GELU(),
        )
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate_act = TemperatureSigmoid(1.5)
        self.highres_fuse = nn.Sequential(
            nn.Conv2d(channels + num_boundaries, channels, kernel_size=1, bias=False),
            _make_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels // 4, bias=False),
            _make_norm(channels),
            nn.GELU(),
        )
        self.cross_scale_gate_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.cross_scale_gate_act = TemperatureSigmoid(1.5)

    def forward(
        self,
        f_edge_refined: torch.Tensor,
        f_region: torch.Tensor,
        b_shared: torch.Tensor,
        b_inner: torch.Tensor,
        b_disjoint: torch.Tensor,
        f_fuse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundary_map = {'intersection': b_shared, 'containment': b_inner, 'disjoint': b_disjoint}

        # High-res path: edge features + boundary priors at edge scale (H/4)
        edge_size = f_edge_refined.shape[-2:]
        boundary_hr = [F.interpolate(boundary_map[r], size=edge_size, mode='bilinear', align_corners=False) for r in self.relation_types]
        edge_detail = self.highres_fuse(torch.cat([f_edge_refined] + boundary_hr, dim=1))

        # Low-res path: all inputs at f_fuse scale (H/16)
        target_size = f_fuse.shape[-2:]
        edge_feature = F.interpolate(f_edge_refined, size=target_size, mode='bilinear', align_corners=False)
        region_feature = F.interpolate(f_region, size=target_size, mode='bilinear', align_corners=False)

        prior_features = [
            self.edge_proj(edge_feature),
            self.region_proj(region_feature),
        ]
        for name in self.relation_types:
            b = F.interpolate(boundary_map[name], size=target_size, mode='bilinear', align_corners=False)
            b = b / (b.abs().mean(dim=(-2, -1), keepdim=True) + 1e-6)
            base = self.boundary_base_proj(b)
            prior_features.append(self.boundary_adapters[name](base))
        prior_features.append(self.fuse_proj(f_fuse))

        if self.prior_attention is not None:
            prior_features = self.prior_attention(prior_features)
        fused = torch.cat(prior_features, dim=1)
        fused = self.fuse(fused)

        # Cross-scale interaction
        fused_up = F.interpolate(fused, size=edge_size, mode='bilinear', align_corners=False)
        cross_gate = self.cross_scale_gate_act(self.cross_scale_gate_conv(torch.cat([edge_detail, fused_up], dim=1)))
        edge_detail = edge_detail * cross_gate

        erga_gate = self.gate_act(self.gate_conv(fused))
        self._last_gate_activations = [erga_gate, cross_gate]
        bottleneck = f_fuse + fused * erga_gate
        return bottleneck, edge_detail

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key_map = {
            'shared_adapter': 'boundary_adapters.intersection',
            'inner_adapter': 'boundary_adapters.containment',
            'disjoint_adapter': 'boundary_adapters.disjoint',
        }
        for old, new in key_map.items():
            for k in list(state_dict.keys()):
                if k.startswith(prefix + old):
                    state_dict[k.replace(prefix + old, prefix + new)] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LowRankChannelAttention(nn.Module):
    def __init__(self, channels: int, rank: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_u = nn.Linear(channels, rank)
        self.fc_v = nn.Linear(channels, rank)
        self.basis = nn.Parameter(torch.randn(rank, channels) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        coeff = self.fc_u(pooled) * self.fc_v(pooled)
        logits = coeff @ self.basis
        return torch.sigmoid(logits).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class LightweightAttention(nn.Module):
    def __init__(self, channels: int, rank: int) -> None:
        super().__init__()
        self.channel = LowRankChannelAttention(channels, rank)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel(x) * self.spatial(x)


class HighLevelTransformer(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[WindowAttentionBlock(channels, num_heads, window_size) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
