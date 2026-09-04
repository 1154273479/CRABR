"""Comparison visualization utilities for CRABR."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
        [0, 128, 255],
    ],
    dtype=np.uint8,
)


def normalize_task_mode(mode: str) -> str:
    return {"binary": "exclusive", "multiclass": "exclusive", "multilabel": "independent"}.get(mode, mode)


def build_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    alpha: float,
    mode: str,
    palette: np.ndarray | None = None,
) -> np.ndarray:
    """Build overlay of image and prediction."""
    gray = image.astype(np.uint8)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    pred_color = prediction_to_color(prediction, mode, palette)
    pred_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(gray_rgb, 1.0 - alpha, pred_bgr, alpha, 0)


def build_comparison_panel(
    image: np.ndarray,
    prediction: np.ndarray,
    mode: str,
    alpha: float,
    ground_truth: np.ndarray | None = None,
    palette: np.ndarray | None = None,
) -> np.ndarray:
    """Build comparison panel with image, ground truth, and prediction."""
    gray = image.astype(np.uint8)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()

    pred_color = cv2.cvtColor(prediction_to_color(prediction, mode, palette), cv2.COLOR_RGB2BGR)
    pred_overlay = build_overlay(gray, prediction, alpha, mode, palette)
    panels = [gray_bgr, pred_color, pred_overlay]

    if ground_truth is not None:
        gt_color = cv2.cvtColor(prediction_to_color(ground_truth, mode, palette), cv2.COLOR_RGB2BGR)
        gt_overlay = build_overlay(gray, ground_truth, alpha, mode, palette)
        panels = [gray_bgr, gt_color, pred_color, gt_overlay, pred_overlay]

    labeled_panels = []
    titles = (
        ["Image", "Prediction", "Overlay"]
        if ground_truth is None
        else ["Image", "GT", "Prediction", "GT Overlay", "Pred Overlay"]
    )
    for title, panel in zip(titles, panels):
        canvas = panel.copy()
        cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        labeled_panels.append(canvas)

    return np.concatenate(labeled_panels, axis=1)


def save_overlay(path: str | Path, overlay: np.ndarray) -> None:
    """Save overlay image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def save_panel(path: str | Path, panel: np.ndarray) -> None:
    """Save comparison panel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)
