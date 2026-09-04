from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encoder import _make_norm
import torch.nn.functional as F


class RelationAwareOverlapPrior(nn.Module):
    """Builds relation priors from region responses.

    Supports configurable relation types and generic statistical features
    that work across arbitrary X-ray datasets (not just chest).
    """

    def __init__(
        self,
        region_num_classes: int,
        region_mode: str,
        task_mode: str,
        relation_types: tuple[str, ...] = ('disjoint', 'intersection', 'containment'),
        use_generic_features: bool = True,
    ) -> None:
        super().__init__()
        self.region_num_classes = region_num_classes
        self.region_mode = region_mode
        self.task_mode = task_mode
        self.relation_types = list(relation_types)
        self.num_relations = len(relation_types)
        self.use_generic_features = use_generic_features

        self.gradient_conv = nn.Conv2d(
            region_num_classes, region_num_classes, kernel_size=3, padding=1,
            groups=region_num_classes, bias=False,
        )
        nn.init.constant_(self.gradient_conv.weight, 0)
        sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        for i in range(region_num_classes):
            self.gradient_conv.weight.data[i, 0] = sobel
        self.gradient_conv.weight.requires_grad_(False)

        feature_dim = 4 + region_num_classes + 1
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 48, kernel_size=3, padding=1, bias=False),
            _make_norm(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=2, dilation=2, bias=False),
            _make_norm(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=1, bias=False),
            _make_norm(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, self.num_relations, kernel_size=1),
        )

    def _spatial_features(self, region_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient magnitude and local variance as spatial features."""
        grad = self.gradient_conv(region_logits)
        grad_mag = grad.abs().sum(dim=1, keepdim=True)
        local_var = F.avg_pool2d(region_logits ** 2, 3, stride=1, padding=1) - F.avg_pool2d(region_logits, 3, stride=1, padding=1) ** 2
        local_var = local_var.mean(dim=1, keepdim=True).clamp(min=0)
        return grad.abs(), local_var

    def _single_label_features(self, region_logits: torch.Tensor, a_region: torch.Tensor) -> torch.Tensor:
        if self.task_mode in {'binary', 'exclusive'} and region_logits.shape[1] <= 2:
            if region_logits.shape[1] == 1:
                fg_prob = torch.sigmoid(region_logits)
                bg_prob = 1.0 - fg_prob
                probs = torch.cat([bg_prob, fg_prob], dim=1)
            else:
                probs = torch.softmax(region_logits, dim=1)
                fg_prob = probs[:, 1:2] if probs.shape[1] > 1 else probs[:, 0:1]
                bg_prob = probs[:, :1]
            competition = 1.0 - (fg_prob - bg_prob).abs()
            uncertainty = 1.0 - torch.abs(2.0 * fg_prob - 1.0)
            disjoint = (1.0 - fg_prob) * (1.0 - competition)
            return torch.cat([fg_prob, competition.clamp(0, 1), disjoint.clamp(0, 1), uncertainty.clamp(0, 1)], dim=1)

        probs = a_region if a_region.shape[1] > 1 else torch.softmax(region_logits, dim=1)
        topk = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
        top1 = topk[:, 0:1]
        top2 = topk[:, 1:2] if probs.shape[1] > 1 else torch.zeros_like(top1)
        competition = (1.0 - (top1 - top2)).clamp(0, 1)
        entropy = -(probs.clamp_min(1e-6) * probs.clamp_min(1e-6).log()).sum(dim=1, keepdim=True)
        entropy = entropy / max(math.log(max(probs.shape[1], 2)), 1.0)
        disjoint = (1.0 - top1) * (1.0 - competition)
        return torch.cat([top1, competition, disjoint.clamp(0, 1), entropy.clamp(0, 1)], dim=1)

    def _multi_label_features(self, region_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(region_logits)
        mean_prob = probs.mean(dim=1, keepdim=True)
        if probs.shape[1] > 1:
            pairwise = (probs.sum(dim=1, keepdim=True).pow(2) - probs.pow(2).sum(dim=1, keepdim=True))
            pairwise = pairwise / float(probs.shape[1] * (probs.shape[1] - 1))
            max_prob = probs.max(dim=1, keepdim=True).values
            residual = (probs.sum(dim=1, keepdim=True) - max_prob) / float(probs.shape[1] - 1)
            disjoint = (1.0 - pairwise).clamp(0, 1) * (1.0 - (max_prob * (1.0 - residual)).clamp(0, 1))
        else:
            pairwise = probs * (1.0 - probs)
            max_prob = probs
            residual = 1.0 - probs
            disjoint = pairwise
        containment = (max_prob * (1.0 - residual)).clamp(0, 1)
        return torch.cat([mean_prob, pairwise.clamp(0, 1), disjoint.clamp(0, 1), containment], dim=1)

    def forward(self, region_logits: torch.Tensor, a_region: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.region_mode == 'single_label':
            stat_features = self._single_label_features(region_logits, a_region)
        elif self.region_mode == 'multi_label':
            stat_features = self._multi_label_features(region_logits)
        else:
            raise ValueError(f'Unsupported region_mode: {self.region_mode}')

        if self.use_generic_features:
            grad_features, local_var = self._spatial_features(region_logits)
            features = torch.cat([stat_features, grad_features, local_var], dim=1)
        else:
            grad_features, local_var = self._spatial_features(region_logits)
            features = torch.cat([stat_features, grad_features, local_var], dim=1)

        priors = self.decoder(features)
        outputs = []
        for i in range(self.num_relations):
            outputs.append(torch.sigmoid(priors[:, i:i+1]))

        if self.num_relations == 3:
            return outputs[0], outputs[1], outputs[2]
        while len(outputs) < 3:
            outputs.append(torch.zeros_like(outputs[0]))
        return outputs[0], outputs[1], outputs[2]
