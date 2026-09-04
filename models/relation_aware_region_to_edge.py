from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationAwareRegionToEdgeConstraint(nn.Module):
    """Relation-aware region-to-edge constraint with gating mechanism.

    Three spatial relations:
    - Disjoint: Two regions are separate, edge should be suppressed
    - Intersection: Two regions intersect, shared boundary should be enhanced
    - Containment: One region contains another, internal contours should be refined

    Gating mechanism allows the model to learn which relations are important.
    """

    RELATION_MODES = {'disjoint': 'suppress', 'intersection': 'enhance', 'containment': 'enhance'}

    def __init__(
        self,
        edge_classes: int,
        region_classes: int,
        edge_channels: int,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.3,
        active_relations: set[str] | None = None,
        containment_scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.active_relations = active_relations if active_relations is not None else {'disjoint', 'intersection', 'containment'}
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.containment_scale = containment_scale
        gate_input_ch = edge_classes + region_classes
        self.gates = nn.ModuleDict()
        for rel in sorted(self.active_relations):
            self.gates[rel] = nn.Sequential(nn.Conv2d(gate_input_ch, 1, kernel_size=1, bias=False), nn.Sigmoid())
        num_rel = len(self.active_relations)
        self.refine_conv = nn.Conv2d(edge_classes * (1 + num_rel) + region_classes, edge_classes, kernel_size=1)
        self.feature_gate_proj = nn.Conv2d(edge_classes, edge_channels, kernel_size=1)

    def forward(
        self,
        f_edge: torch.Tensor,
        a_edge: torch.Tensor,
        a_region: torch.Tensor,
        o_disjoint: torch.Tensor,
        o_intersection: torch.Tensor,
        o_containment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a_region_up = F.interpolate(a_region, size=a_edge.shape[-2:], mode='bilinear', align_corners=False)
        o_disjoint_up = F.interpolate(o_disjoint, size=a_edge.shape[-2:], mode='bilinear', align_corners=False)
        o_inter_up = F.interpolate(o_intersection, size=a_edge.shape[-2:], mode='bilinear', align_corners=False)
        o_cont_up = F.interpolate(o_containment, size=a_edge.shape[-2:], mode='bilinear', align_corners=False)

        gate_input = torch.cat([a_edge, a_region_up], dim=1)

        relation_cues = {'disjoint': o_disjoint_up, 'intersection': o_inter_up, 'containment': o_cont_up}
        weight_map = {'disjoint': self.gamma, 'intersection': self.alpha, 'containment': self.beta}

        branch_outputs = []
        a_edge_disjoint = a_edge
        a_edge_inter = a_edge
        a_edge_inner = a_edge

        for rel in sorted(self.active_relations):
            g = self.gates[rel](gate_input)
            o = relation_cues.get(rel, torch.zeros_like(a_edge[:, :1]))
            w = weight_map.get(rel, self.alpha)
            mode = self.RELATION_MODES.get(rel, 'enhance')

            if rel == 'containment':
                g = g * self.containment_scale

            if mode == 'suppress':
                branch = a_edge * (1.0 - w * o * g)
            else:
                branch = a_edge * (1.0 + w * o * g)
            branch_outputs.append(branch)

            if rel == 'disjoint':
                a_edge_disjoint = branch
            elif rel == 'intersection':
                a_edge_inter = branch
            elif rel == 'containment':
                a_edge_inner = branch

        a_edge_refined = self.refine_conv(torch.cat([a_edge] + branch_outputs + [a_region_up], dim=1))

        feature_gate = F.interpolate(a_edge_refined, size=f_edge.shape[-2:], mode='bilinear', align_corners=False)
        feature_gate = torch.sigmoid(self.feature_gate_proj(feature_gate))
        f_edge_refined = f_edge * feature_gate
        return f_edge_refined, a_edge_refined, a_edge_disjoint, a_edge_inter, a_edge_inner

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key_map = {
            'gate_disjoint': 'gates.disjoint',
            'gate_inter': 'gates.intersection',
            'gate_inner': 'gates.containment',
        }
        for old, new in key_map.items():
            for k in list(state_dict.keys()):
                if k.startswith(prefix + old):
                    state_dict[k.replace(prefix + old, prefix + new)] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class RelationAwareEdgeToRegionConstraint(nn.Module):
    """Edge-to-region feedback constraint.

    This module complements RelationAwareRegionToEdgeConstraint. It uses the
    original edge attention and relation boundary priors to update region
    attention and region features in parallel with the region-to-edge path.
    """

    def __init__(
        self,
        edge_classes: int,
        region_classes: int,
        region_channels: int,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.3,
        active_relations: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.active_relations = active_relations if active_relations is not None else {'disjoint', 'intersection', 'containment'}
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        gate_input_ch = edge_classes + region_classes
        self.gate_boundary = nn.Sequential(nn.Conv2d(gate_input_ch, 1, kernel_size=1, bias=False), nn.Sigmoid())
        self.refine_conv = nn.Conv2d(region_classes * 2 + edge_classes + 3, region_classes, kernel_size=1)
        self.feature_gate_proj = nn.Conv2d(region_classes, region_channels, kernel_size=1)

    def forward(
        self,
        f_region: torch.Tensor,
        a_region: torch.Tensor,
        a_edge: torch.Tensor,
        b_shared: torch.Tensor,
        b_inner: torch.Tensor,
        b_disjoint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_size = a_region.shape[-2:]
        a_edge_up = F.interpolate(a_edge, size=target_size, mode='bilinear', align_corners=False)
        b_shared_up = F.interpolate(b_shared, size=target_size, mode='bilinear', align_corners=False)
        b_inner_up = F.interpolate(b_inner, size=target_size, mode='bilinear', align_corners=False)
        b_disjoint_up = F.interpolate(b_disjoint, size=target_size, mode='bilinear', align_corners=False)

        gate = self.gate_boundary(torch.cat([a_edge_up, a_region], dim=1))
        boundary_boost = 1.0
        if 'intersection' in self.active_relations:
            boundary_boost = boundary_boost + self.alpha * b_shared_up * gate
        if 'containment' in self.active_relations:
            boundary_boost = boundary_boost + self.beta * b_inner_up * gate
        if 'disjoint' in self.active_relations:
            boundary_boost = boundary_boost - self.gamma * b_disjoint_up * gate

        a_region_boundary = a_region * boundary_boost
        a_region_refined = self.refine_conv(
            torch.cat(
                [a_region, a_region_boundary, a_edge_up, b_shared_up, b_inner_up, b_disjoint_up],
                dim=1,
            )
        )
        feature_gate = F.interpolate(a_region_refined, size=f_region.shape[-2:], mode='bilinear', align_corners=False)
        feature_gate = torch.sigmoid(self.feature_gate_proj(feature_gate))
        f_region_refined = f_region * feature_gate
        return f_region_refined, a_region_refined
