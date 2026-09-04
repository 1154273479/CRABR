from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNormAct, _make_norm


def _make_laplacian_3x3() -> torch.Tensor:
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    return kernel.unsqueeze(0).unsqueeze(0)


def _make_sobel_x() -> torch.Tensor:
    kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    return kernel.unsqueeze(0).unsqueeze(0)


def _make_sobel_y() -> torch.Tensor:
    kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    return kernel.unsqueeze(0).unsqueeze(0)


class CurvatureAwareBoundaryRefinement(nn.Module):
    """Extract curvature and normal direction from SDM predictions to refine decoder features.

    Uses fixed Laplacian/Sobel kernels to compute geometric features (curvature, normals),
    then learns a spatial gate that modulates decoder features based on boundary geometry.
    """

    def __init__(self, num_classes: int, decoder_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('laplacian_kernel', _make_laplacian_3x3().repeat(num_classes, 1, 1, 1))
        self.register_buffer('sobel_x', _make_sobel_x().repeat(num_classes, 1, 1, 1))
        self.register_buffer('sobel_y', _make_sobel_y().repeat(num_classes, 1, 1, 1))

        self.geo_proj = nn.Sequential(
            ConvNormAct(num_classes * 4, decoder_dim, kernel_size=1),
            ConvNormAct(decoder_dim, decoder_dim, kernel_size=3),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, sdm_pred: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        C = self.num_classes
        curvature = F.conv2d(sdm_pred, self.laplacian_kernel, padding=1, groups=C)
        normal_x = F.conv2d(sdm_pred, self.sobel_x, padding=1, groups=C)
        normal_y = F.conv2d(sdm_pred, self.sobel_y, padding=1, groups=C)
        geo_feat = torch.cat([curvature, curvature.abs(), normal_x, normal_y], dim=1)
        geo_feat = self.geo_proj(geo_feat)
        gate = self.gate(geo_feat)
        return decoder_feat + geo_feat * gate


class InterClassDistanceConsistency(nn.Module):
    """Compute geometric relation priors from inter-class SDM distance relationships.

    For each class pair, computes absolute distance and sign agreement between SDMs,
    then predicts geometry-based relation maps (disjoint/intersection/containment).
    """

    def __init__(self, num_classes: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.num_classes = num_classes
        num_pairs = num_classes * (num_classes - 1) // 2
        in_ch = num_pairs * 2
        self.relation_net = nn.Sequential(
            ConvNormAct(in_ch, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            _make_norm(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, sdm_pred: torch.Tensor) -> torch.Tensor:
        C = self.num_classes
        pairs_dist: list[torch.Tensor] = []
        pairs_sign: list[torch.Tensor] = []
        for i in range(C):
            for j in range(i + 1, C):
                pairs_dist.append((sdm_pred[:, i:i+1] - sdm_pred[:, j:j+1]).abs())
                pairs_sign.append(sdm_pred[:, i:i+1] * sdm_pred[:, j:j+1])
        feat = torch.cat(pairs_dist + pairs_sign, dim=1)
        return self.relation_net(feat)
