from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np

from utils.config import load_config
from visualize_prediction_matrices import (
    REPO_ROOT,
    build_sample_lookup,
    class_items_for_mask,
    discover_overlay_paths,
    finish_tile,
    normalize_to_bgr,
    pick_sample_ids,
    resize_to_match,
)


DATASET_SPECS = {
    "jsrt_cxr": {
        "display_name": "JSRT_CXR",
        "size_to_config": {
            "512": "configs/jsrt_scr.yaml",
            "256": "configs/jsrt_scr_512_nopretrain.yaml",
            "384": "configs/jsrt_scr_384_nopretrain.yaml",
        },
        "size_to_dataset_key": {
            "512": "jsrt_scr_512_nopretrain",
            "256": "jsrt_scr_512_nopretrain_256",
            "384": "jsrt_scr_384_nopretrain_384",
        },
    },
    "vindr_ribcxr": {
        "display_name": "VinDr-RibCXR",
        "size_to_config": {
            "512": "configs/vindr_rib.yaml",
            "256": "configs/vindr_rib.yaml",
            "384": "configs/vindr_rib_384.yaml",
        },
        "size_to_dataset_key": {
            "512": "vindr_rib",
            "256": "vindr_rib_256",
            "384": "vindr_rib_384",
        },
    },
}

MODEL_ORDER = [
    "erga",
    "swin_unet",
    "u_mamba",
    "vm_unet_v2",
    "segmamba",
    "msvm_unet",
    "i2u_net",
    "dcm_net",
    "tbconvl_net",
    "cfm_unet",
    "kmunet",
]

MODEL_LABELS = {
    "image_gt": "Image+GT",
    "erga": "CRABR",
    "swin_unet": "Swin-UNet",
    "u_mamba": "U-Mamba",
    "vm_unet_v2": "VM-UNet-V2",
    "segmamba": "SegMamba",
    "msvm_unet": "MSVM-UNet",
    "i2u_net": "I2U-Net",
    "dcm_net": "DCM-Net",
    "tbconvl_net": "TBConvL-Net",
    "cfm_unet": "CFM-UNet",
    "kmunet": "KM-UNet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate custom comparison figures for JSRT_CXR and VinDr-RibCXR.")
    parser.add_argument("--size", choices=["256", "384", "512"], default="512", help="Visualization size tag.")
    parser.add_argument("--samples-per-dataset", type=int, default=5, help="Number of samples per dataset.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "visualizations" / "custom_comparisons"),
        help="Directory to save generated figures.",
    )
    return parser.parse_args()


def make_placeholder(width: int, height: int) -> np.ndarray:
    return finish_tile(np.full((height, width, 3), 240, dtype=np.uint8))


def render_image_gt_tile(image_bgr: np.ndarray, mask: np.ndarray, class_names: list[str]) -> np.ndarray:
    canvas = image_bgr.copy()
    overlay = image_bgr.copy()
    vivid_colors = [
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 80, 0),
        (0, 200, 0),
    ]
    for idx, (_, binary, _) in enumerate(class_items_for_mask(mask, class_names)):
        color = vivid_colors[idx % len(vivid_colors)]
        overlay[binary > 0] = color
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 5, lineType=cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.46, canvas, 0.54, 0.0)
    return finish_tile(canvas)


def add_horizontal_gap(tiles: list[np.ndarray], gap: int, filler: int = 255) -> np.ndarray:
    row = tiles[0]
    for tile in tiles[1:]:
        spacer = np.full((row.shape[0], gap, 3), filler, dtype=np.uint8)
        row = np.concatenate([row, spacer, tile], axis=1)
    return row


def assemble_rows(rows: list[np.ndarray], gap: int, filler: int = 255) -> np.ndarray:
    canvas = rows[0]
    for row in rows[1:]:
        spacer = np.full((gap, canvas.shape[1], 3), filler, dtype=np.uint8)
        canvas = np.concatenate([canvas, spacer, row], axis=0)
    return canvas


def split_label_text(text: str) -> list[str]:
    if len(text) <= 11:
        return [text]
    if "+" in text:
        return text.split("+", 1)
    if "-" in text:
        parts = text.split("-")
        if len(parts) == 2:
            return parts
        return ["-".join(parts[:-1]), parts[-1]]
    camel_parts = re.findall(r"[A-Z]+[a-z0-9]*|[A-Z]?[a-z0-9]+", text)
    if len(camel_parts) >= 2:
        mid = len(camel_parts) // 2
        return ["".join(camel_parts[:mid]), "".join(camel_parts[mid:])]
    return [text]


def make_label_tile(text: str, width: int, height: int) -> np.ndarray:
    tile = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    lines = split_label_text(text)
    scale = 1.45 if len(lines) == 1 else 1.2
    thickness = 4
    line_gap = 18
    metrics = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    total_height = sum(text_h for (_text_w, text_h), _baseline in metrics)
    total_height += line_gap * (len(lines) - 1)
    current_y = max((height - total_height) // 2, 10)
    for line, ((text_w, text_h), baseline) in zip(lines, metrics):
        x = max((width - text_w) // 2, 10)
        y = current_y + text_h
        cv2.putText(tile, line, (x, y), font, scale, (10, 10, 10), thickness, lineType=cv2.LINE_AA)
        current_y = y + baseline + line_gap
    return tile


def build_row_tiles(
    image_id: str,
    sample_lookup: dict[str, dict[str, np.ndarray]],
    overlay_index: dict[str, dict[str, Path]],
    class_names: list[str],
) -> list[np.ndarray]:
    raw_image = normalize_to_bgr(sample_lookup[image_id]["raw_image"])
    gt_mask = sample_lookup[image_id]["mask"]
    base_h, base_w = raw_image.shape[:2]
    image_gt_tile = render_image_gt_tile(raw_image, gt_mask, class_names)
    tiles = [image_gt_tile]

    for model_name in MODEL_ORDER:
        overlay_path = overlay_index.get(model_name, {}).get(image_id)
        if overlay_path is None:
            tiles.append(make_placeholder(base_w, base_h))
            continue
        overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if overlay is None:
            tiles.append(make_placeholder(base_w, base_h))
            continue
        tiles.append(finish_tile(resize_to_match(overlay, base_w, base_h)))
    return tiles


def build_figure_for_dataset(dataset_id: str, spec: dict[str, dict[str, str]], size_tag: str, sample_count: int) -> tuple[np.ndarray, dict]:
    cfg_path = (REPO_ROOT / spec["size_to_config"][size_tag]).resolve()
    cfg = load_config(str(cfg_path))
    cfg["data"]["image_size"] = [int(size_tag), int(size_tag)]
    dataset_key = spec["size_to_dataset_key"][size_tag]

    overlay_index = discover_overlay_paths(size_tag, dataset_key)
    missing_models = [model_name for model_name in MODEL_ORDER if model_name not in overlay_index]
    if missing_models:
        raise FileNotFoundError(f"Missing overlays for {dataset_id}: {missing_models}")

    sample_lookup = build_sample_lookup(cfg)
    sample_ids = pick_sample_ids(overlay_index, sample_lookup, sample_count)
    if not sample_ids:
        raise RuntimeError(f"No matched samples found for {dataset_id}")

    gap = 12
    class_names = list(cfg["data"].get("class_names", []))
    row_images = [
        add_horizontal_gap(build_row_tiles(image_id, sample_lookup, overlay_index, class_names), gap)
        for image_id in sample_ids
    ]
    label_keys = ["image_gt"] + MODEL_ORDER
    sample_tile_w = normalize_to_bgr(sample_lookup[sample_ids[0]]["raw_image"]).shape[1]
    label_tiles = [make_label_tile(MODEL_LABELS[key], sample_tile_w, 170) for key in label_keys]
    label_row = add_horizontal_gap(label_tiles, gap)

    figure = assemble_rows(row_images + [label_row], gap)
    metadata = {
        "dataset_id": dataset_id,
        "display_name": spec["display_name"],
        "size_tag": size_tag,
        "config_path": str(cfg_path),
        "dataset_key": dataset_key,
        "sample_ids": sample_ids,
        "model_order": MODEL_ORDER,
        "model_labels": [MODEL_LABELS["image_gt"]] + [MODEL_LABELS[name] for name in MODEL_ORDER],
    }
    return figure, metadata


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve() / args.size
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_id, spec in DATASET_SPECS.items():
        figure, metadata = build_figure_for_dataset(dataset_id, spec, args.size, args.samples_per_dataset)
        image_path = output_root / f"{dataset_id}_comparison.png"
        meta_path = output_root / f"{dataset_id}_comparison.json"
        cv2.imwrite(str(image_path), figure)
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Saved {image_path}")


if __name__ == "__main__":
    main()
