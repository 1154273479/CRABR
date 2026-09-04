"""Ablation experiment groups for CRABR model.

3-stage progressive ablation:
    Stage 1: Baseline (minimal ERGA)
    Stage 2: + FE + HCLF + DualBranch
    Stage 3: + GeoLoop
    Stage 4: + ERGA (full model)
"""
from __future__ import annotations

from typing import Any, Dict, List


def _loss_zeros(*keys: str) -> Dict[str, float]:
    return {key: 0.0 for key in keys}


# =============================================================================
# Loss Keys
# =============================================================================
STRUCTURE_LOSS_KEYS = (
    "lambda_edge",
    "lambda_shared",
    "lambda_inner",
    "lambda_disjoint",
    "lambda_cons",
    "lambda_boundary_dist",
    "lambda_boundary_focal",
    "lambda_sdm",
    "lambda_coarse_sdm",
    "lambda_curvature",
    "lambda_direction",
    "lambda_topology",
    "lambda_adjacency",
    "lambda_edge_consistency",
    "lambda_uncertainty",
    "lambda_unc_seg",
    "lambda_gate_entropy",
    "lambda_gate_diversity",
)

UNCERTAINTY_LOSS_KEYS = ("lambda_uncertainty", "lambda_unc_seg")

# =============================================================================
# Model Configurations
# =============================================================================
# Stage 1: Baseline - minimal ERGA without most modules
STAGE1_MODEL = {
    "disable_feature_enhancement": True,
    "cross_level_fusion": "f3_only",
    "attention_branches": "none",
    "relation_guidance": "none",
    "disable_boundary_generators": True,
    "disable_coarse_sdm": True,
    "disable_icdc": True,
    "disable_erga": True,
    "bottleneck_enhancement": "none",
    "disable_uncertainty_head": True,
    "disable_curriculum_gradient": True,
}

# Stage 1: Base loss
STAGE1_LOSS = {
    "lambda_ce": 1.0,
    "lambda_dice": 1.0,
    "lambda_router": 0.0,
    **_loss_zeros(*STRUCTURE_LOSS_KEYS),
}

# Stage 2: + Feature Enhancement + HCLF + Dual Branch
STAGE2_MODEL = {
    "disable_feature_enhancement": False,
    "cross_level_fusion": "gated",
    "attention_branches": "both",
    "relation_guidance": "none",
    "disable_boundary_generators": True,
    "disable_coarse_sdm": True,
    "disable_icdc": True,
    "disable_erga": True,
    "bottleneck_enhancement": "none",
    "disable_uncertainty_head": True,
    "disable_curriculum_gradient": True,
}

STAGE2_LOSS = {
    "lambda_ce": 1.0,
    "lambda_dice": 1.0,
    "lambda_router": 0.0,
    "lambda_edge": 0.2,
    **_loss_zeros(
        "lambda_shared",
        "lambda_inner",
        "lambda_disjoint",
        "lambda_cons",
        "lambda_boundary_dist",
        "lambda_boundary_focal",
        "lambda_sdm",
        "lambda_coarse_sdm",
        "lambda_curvature",
        "lambda_direction",
        "lambda_topology",
        "lambda_adjacency",
        "lambda_edge_consistency",
        "lambda_uncertainty",
        "lambda_unc_seg",
        "lambda_gate_entropy",
        "lambda_gate_diversity",
    ),
}

# Stage 3: + Geometry Loop (coarse SDM + ICDC + curriculum)
STAGE3_MODEL = {
    "disable_feature_enhancement": False,
    "cross_level_fusion": "gated",
    "attention_branches": "both",
    "relation_guidance": "full",
    "disable_boundary_generators": False,
    "disable_coarse_sdm": False,
    "disable_icdc": False,
    "disable_erga": True,
    "bottleneck_enhancement": "none",
    "disable_uncertainty_head": True,
    "disable_curriculum_gradient": False,
}

STAGE3_LOSS = {
    "lambda_ce": 1.0,
    "lambda_dice": 1.0,
    "lambda_router": 0.0,
    "lambda_edge": 0.2,
    "lambda_shared": 0.05,
    "lambda_inner": 0.02,
    "lambda_disjoint": 0.05,
    "lambda_cons": 0.02,
    "lambda_sdm": 0.5,
    "lambda_coarse_sdm": 0.2,
    "lambda_direction": 0.05,
    **_loss_zeros(
        "lambda_boundary_dist",
        "lambda_boundary_focal",
        "lambda_curvature",
        "lambda_topology",
        "lambda_adjacency",
        "lambda_edge_consistency",
        "lambda_uncertainty",
        "lambda_unc_seg",
        "lambda_gate_entropy",
        "lambda_gate_diversity",
    ),
}

# Stage 4: + ERGA (Full model)
STAGE4_MODEL = {
    "disable_feature_enhancement": False,
    "cross_level_fusion": "gated",
    "attention_branches": "both",
    "relation_guidance": "full",
    "disable_boundary_generators": False,
    "disable_coarse_sdm": False,
    "disable_icdc": False,
    "disable_erga": False,
    "bottleneck_enhancement": "full",
    "disable_uncertainty_head": False,
    "disable_curriculum_gradient": False,
}

STAGE4_LOSS = {
    "lambda_ce": 1.0,
    "lambda_dice": 1.0,
    "lambda_router": 0.0,
    "lambda_edge": 0.2,
    "lambda_shared": 0.05,
    "lambda_inner": 0.02,
    "lambda_disjoint": 0.05,
    "lambda_cons": 0.02,
    "lambda_sdm": 0.5,
    "lambda_coarse_sdm": 0.2,
    "lambda_direction": 0.05,
    "lambda_boundary_dist": 0.1,
    "lambda_boundary_focal": 1.0,
    "lambda_gate_entropy": 0.0003,
    **_loss_zeros(
        "lambda_curvature",
        "lambda_topology",
        "lambda_adjacency",
        "lambda_edge_consistency",
        "lambda_uncertainty",
        "lambda_unc_seg",
    ),
}

# =============================================================================
# Dataset Labels
# =============================================================================
DATASET_LABELS = {
    "jsrt_scr": "JSRT",
    "vindr_rib": "VinDr-Rib",
}

# =============================================================================
# Ablation Groups
# =============================================================================
DEFAULT_CONFIGS = [
    "configs/jsrt_scr.yaml",
    "configs/vindr_rib.yaml",
]

DEFAULT_SEEDS = [42, 1337, 2025]

GROUP_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "main_modules": [
        {
            "name": "baseline",
            "label": "Baseline",
            "stage": 1,
            "model_overrides": STAGE1_MODEL,
            "loss_overrides": STAGE1_LOSS,
        },
        {
            "name": "plus_fe_hclf_branch",
            "label": "+FE+HCLF+Branch",
            "stage": 2,
            "model_overrides": STAGE2_MODEL,
            "loss_overrides": STAGE2_LOSS,
        },
        {
            "name": "plus_geoloop",
            "label": "+GeoLoop",
            "stage": 3,
            "model_overrides": STAGE3_MODEL,
            "loss_overrides": STAGE3_LOSS,
        },
        {
            "name": "full",
            "label": "Full",
            "stage": 4,
            "model_overrides": STAGE4_MODEL,
            "loss_overrides": STAGE4_LOSS,
        },
    ],
}
