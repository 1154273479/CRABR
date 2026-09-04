"""Visualization utilities for CRABR."""
from __future__ import annotations

from .comparison import build_comparison_panel, build_overlay, save_overlay, save_panel
from .prediction_matrices import visualize_prediction_matrices
from .palette import build_palette, palette_from_config, multilabel_to_color

__all__ = [
    "build_comparison_panel",
    "build_overlay",
    "save_overlay",
    "save_panel",
    "visualize_prediction_matrices",
    "build_palette",
    "palette_from_config",
    "multilabel_to_color",
]
