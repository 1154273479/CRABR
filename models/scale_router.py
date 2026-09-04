from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleRouter(nn.Module):
    """Predicts per-class scale routing weights from bottleneck features.

    Given the bottleneck (which encodes region/edge/relation semantics),
    predicts how much each class should attend to each encoder scale.
    Small targets learn to prefer high-resolution scales (F1/F2),
    large targets prefer low-resolution scales (F3/F4).
    """

    def __init__(self, channels: int, num_scales: int = 4, num_classes: int = 6, temperature: float = 0.5) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.num_classes = num_classes
        self.temperature = temperature
        mid = max(channels // 4, 32)
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, num_classes * num_scales),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Returns scale_weights [B, num_classes, num_scales] with softmax over scales."""
        B = bottleneck.shape[0]
        logits = self.predictor(bottleneck)
        logits = logits.view(B, self.num_classes, self.num_scales)
        return torch.softmax(logits / self.temperature, dim=-1)

    def entropy_loss(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Compute normalized entropy of routing weights (minimize to encourage sparsity)."""
        weights = self.forward(bottleneck)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)  # [B, num_classes]
        max_entropy = math.log(self.num_scales)
        return (entropy / max_entropy).mean()


class ScaleAdaptiveSkip(nn.Module):
    """Multi-scale weighted skip connection.

    Instead of using a single fixed-scale skip (e.g., only F2 for stage2),
    this module blends all encoder scales with learned class-aware weights,
    then collapses the class dimension to produce a single spatial feature map.
    """

    def __init__(self, common_dim: int, num_scales: int = 4) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(common_dim, common_dim, kernel_size=1, bias=False)
            for _ in range(num_scales)
        ])
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        encoder_feats: list[torch.Tensor],
        scale_weights: torch.Tensor,
        default_skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_feats: [f1, f2, f3, f4] each [B, C, H_i, W_i]
            scale_weights: [B, num_classes, num_scales]
            default_skip: the original fixed skip for this stage [B, C, H_t, W_t]
        Returns:
            adaptive_skip: [B, C, H_t, W_t]
        """
        target_size = default_skip.shape[-2:]
        B, C = default_skip.shape[:2]

        # Collapse class dimension: average scale weights across classes → [B, num_scales]
        w = scale_weights.mean(dim=1)  # [B, num_scales]

        adaptive = torch.zeros_like(default_skip)
        for i, (feat, proj) in enumerate(zip(encoder_feats, self.scale_projs)):
            aligned = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            adaptive = adaptive + w[:, i].view(B, 1, 1, 1) * proj(aligned)

        return default_skip + self.alpha * (adaptive - default_skip)
