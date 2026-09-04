from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

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


def parse_color(value: object) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 6:
            return [int(text[idx : idx + 2], 16) for idx in (0, 2, 4)]
        parts = [part.strip() for part in text.split(",")]
        if len(parts) == 3:
            return [int(part) for part in parts]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        parts = list(value)
        if len(parts) == 3:
            return [int(part) for part in parts]
    raise ValueError(f"Unsupported color value: {value!r}")


def build_palette(
    colors: object | None,
    class_names: Sequence[str] | None = None,
    num_classes: int | None = None,
) -> np.ndarray:
    class_names = list(class_names or [])
    target_size = int(num_classes or len(class_names) or len(DEFAULT_PALETTE))
    target_size = max(target_size, 1)
    repeats = int(np.ceil(target_size / len(DEFAULT_PALETTE)))
    palette = np.tile(DEFAULT_PALETTE, (repeats, 1))[:target_size].copy()

    if colors is None:
        return palette

    if isinstance(colors, Mapping):
        for key, value in colors.items():
            if isinstance(key, int) or str(key).isdigit():
                idx = int(key)
            else:
                if str(key) not in class_names:
                    continue
                idx = class_names.index(str(key))
            if 0 <= idx < len(palette):
                palette[idx] = parse_color(value)
        return palette

    if isinstance(colors, Sequence) and not isinstance(colors, (bytes, bytearray, str)):
        for idx, value in enumerate(colors):
            if idx >= len(palette):
                break
            palette[idx] = parse_color(value)
        return palette

    raise ValueError("Colors must be a list or mapping")


def palette_from_config(cfg: dict) -> np.ndarray:
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("inference", {})
    colors = infer_cfg.get("class_colors", infer_cfg.get("palette", data_cfg.get("class_colors")))
    return build_palette(
        colors,
        class_names=data_cfg.get("class_names", []),
        num_classes=int(cfg.get("model", {}).get("num_classes", data_cfg.get("num_classes", 0))),
    )


def prediction_to_color(mask: np.ndarray, mode: str, palette: np.ndarray | None = None) -> np.ndarray:
    mode = normalize_task_mode(mode)
    palette = DEFAULT_PALETTE if palette is None else palette
    if mode == "exclusive" and mask.ndim == 2:
        return label_to_color(mask, palette)
    if mask.ndim == 2:
        binary = (mask > 0).astype(np.uint8)
        color = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        color[binary > 0] = palette[1 % len(palette)]
        return color
    return multilabel_to_color(mask, palette)


def label_to_color(mask: np.ndarray, palette: np.ndarray | None = None) -> np.ndarray:
    palette = DEFAULT_PALETTE if palette is None else palette
    return palette[mask.astype(np.int64) % len(palette)]


def multilabel_to_color(mask: np.ndarray, palette: np.ndarray | None = None) -> np.ndarray:
    palette = DEFAULT_PALETTE if palette is None else palette
    color = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.float32)
    palette = palette.astype(np.float32)
    count = np.zeros((mask.shape[1], mask.shape[2], 1), dtype=np.float32)
    for idx in range(mask.shape[0]):
        region = mask[idx] > 0
        color[region] += palette[idx % len(palette)]
        count[region] += 1
    count = np.maximum(count, 1)
    return (color / count).astype(np.uint8)


def build_overlay(image: np.ndarray, prediction: np.ndarray, alpha: float, mode: str, palette: np.ndarray | None = None) -> np.ndarray:
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
    titles = ["Image", "Prediction", "Overlay"] if ground_truth is None else ["Image", "GT", "Prediction", "GT Overlay", "Pred Overlay"]
    for title, panel in zip(titles, panels):
        canvas = panel.copy()
        cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        labeled_panels.append(canvas)

    return np.concatenate(labeled_panels, axis=1)


def save_prediction_mask(
    path: str | Path,
    mask: np.ndarray,
    mode: str,
    palette: np.ndarray | None = None,
    colorize: bool = False,
    class_index: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_mode = mode
    mode = normalize_task_mode(mode)

    if colorize:
        if class_index is not None and mask.ndim == 2:
            palette = DEFAULT_PALETTE if palette is None else palette
            color = np.zeros((*mask.shape, 3), dtype=np.uint8)
            color[mask > 0] = palette[class_index % len(palette)]
        else:
            color = prediction_to_color(mask, mode, palette)
        cv2.imwrite(str(path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    elif mode == "exclusive" and mask.ndim == 2:
        if original_mode == "binary":
            cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
            return
        cv2.imwrite(str(path), mask.astype(np.uint8))
    elif mask.ndim == 3:
        color = cv2.cvtColor(multilabel_to_color(mask, palette), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), color)
    else:
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def save_overlay(path: str | Path, overlay: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def save_panel(path: str | Path, panel: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)
