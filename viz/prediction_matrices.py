"""Visualization matrix generator for CRABR paper figures.

Creates comparison matrices showing predictions from different models across datasets.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.config import load_config


REPO_ROOT = Path(__file__).parent.parent.resolve()
PREDICTION_ROOT = REPO_ROOT / "predictions"
OUTPUT_ROOT = REPO_ROOT / "visualizations" / "prediction_matrices"
RESULTS_ROOT = REPO_ROOT / "results"

DATASET_SPECS = [
    {
        "dataset_id": "jsrt_scr",
        "display_name": "JSRT-SCR",
        "size_to_config": {
            "512": "configs/jsrt_scr.yaml",
            "256": "configs/jsrt_scr_512_nopretrain.yaml",
        },
        "size_to_dataset_key": {
            "512": "jsrt_scr_512_nopretrain",
            "256": "jsrt_scr_512_nopretrain_256",
        },
    },
    {
        "dataset_id": "vindr_rib",
        "display_name": "VinDr Rib",
        "size_to_config": {
            "512": "configs/vindr_rib.yaml",
            "256": "configs/vindr_rib.yaml",
        },
        "size_to_dataset_key": {
            "512": "vindr_rib",
            "256": "vindr_rib_256",
        },
    },
]

MODEL_DISPLAY_NAMES = {
    "crab": "CRABR",
    "unet": "U-Net",
    "unetplusplus": "U-Net++",
    "swin_unet": "Swin U-Net",
    "swin_transformer": "Swin Transformer",
    "u_mamba": "U-Mamba",
    "vm_unet_v2": "VM-UNet v2",
    "msvm_unet": "MSVM-UNet",
    "segmamba": "SegMamba",
    "medsam2_adapter": "MedSAM2 Adapter",
    "kmunet": "KMUNet",
    "dcm_net": "DCM-Net",
    "cfm_unet": "CFM-UNet",
    "i2u_net": "I2U-Net",
    "tbconvl_net": "TBConvL-Net",
}

MODEL_CHOICES = (
    "crab",
    "unet",
    "unetplusplus",
    "swin_unet",
    "swin_transformer",
    "u_mamba",
    "vm_unet_v2",
    "msvm_unet",
    "segmamba",
    "medsam2_adapter",
    "kmunet",
    "dcm_net",
    "cfm_unet",
    "i2u_net",
    "tbconvl_net",
)

CONTOUR_COLORS_BGR = [
    (60, 179, 113),
    (0, 165, 255),
    (255, 191, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (128, 0, 255),
    (0, 128, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build visualization matrices from saved segmentation overlays."
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=["256", "512"],
        help="Image sizes to export. Default: 256 512",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[item["dataset_id"] for item in DATASET_SPECS],
        help="Dataset ids to export. Default: jsrt_scr vindr_rib",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=5,
        help="Number of images to include per dataset matrix",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT),
        help="Directory to save matrices and metadata",
    )
    parser.add_argument(
        "--top-k-models",
        type=int,
        default=10,
        help="Only keep the top-K models ranked by average Dice across datasets and sizes",
    )
    return parser.parse_args()


def discover_overlay_paths(size_tag: str, dataset_key: str) -> dict[str, dict[str, Path]]:
    """Discover available overlay paths for models."""
    size_root = PREDICTION_ROOT if size_tag == "512" else PREDICTION_ROOT / size_tag
    models = ["crab"] + [name for name in MODEL_CHOICES if name != "crab"]
    overlay_index: dict[str, dict[str, Path]] = {}

    for model_name in models:
        if model_name == "crab":
            dataset_dir = size_root / dataset_key
        else:
            dataset_dir = size_root / model_name / dataset_key
        if not dataset_dir.exists():
            continue

        sample_to_path: dict[str, Path] = {}
        for fold_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for file_path in sorted(fold_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                if not file_path.name.endswith("_overlay.png"):
                    continue
                image_id = file_path.name[: -len("_overlay.png")]
                sample_to_path.setdefault(image_id, file_path)
        if sample_to_path:
            overlay_index[model_name] = sample_to_path

    return overlay_index


def collect_model_scores() -> dict[str, dict[str, float]]:
    """Collect Dice scores for all models."""
    dataset_keys = ["jsrt_scr_512_nopretrain", "vindr_rib"]
    pattern = re.compile(
        r"^(?:(?P<model>.+)_)?(?P<dataset>jsrt_scr_512_nopretrain|vindr_rib)"
        r"(?:_(?P<size>256|384))?_kfold_summary\.json$"
    )
    scores: dict[str, dict[str, float]] = {}

    for result_dir, size_tag in ((RESULTS_ROOT / "256", "256"), (RESULTS_ROOT, "512")):
        if not result_dir.exists():
            continue
        for file_path in sorted(result_dir.glob("*_kfold_summary.json")):
            name = file_path.name
            if name.startswith("dataset_grouped_"):
                continue
            if size_tag == "512" and "pretrain" in name and "jsrt_scr_512_nopretrain" not in name:
                continue
            if size_tag == "512" and ("_256_" in name or "_384_" in name):
                continue

            match = pattern.match(name)
            if not match:
                continue
            matched_dataset = match.group("dataset")
            if matched_dataset not in dataset_keys:
                continue
            model_name = match.group("model") or "crab"
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                dice = float(payload["summary"]["dice"]["mean"])
            except Exception:
                continue
            scores.setdefault(model_name, {})[f"{matched_dataset}_{size_tag}"] = dice
    return scores


def rank_top_models(top_k: int) -> list[dict[str, Any]]:
    """Rank models by average Dice score."""
    scores = collect_model_scores()
    ranked = []
    for model_name, task_scores in scores.items():
        if not task_scores:
            continue
        values = [task_scores[key] for key in sorted(task_scores)]
        ranked.append(
            {
                "model_name": model_name,
                "avg_dice": float(sum(values) / len(values)),
                "task_count": len(values),
                "task_keys": sorted(task_scores),
            }
        )
    ranked.sort(key=lambda item: (-item["avg_dice"], -item["task_count"], item["model_name"]))
    return ranked[:top_k]


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMG_EXTS | {".npy", ".npz"}


def list_files(path: Path, valid_exts: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts)


def stem_aliases(stem: str) -> set[str]:
    aliases = {stem}
    for suffix in ["_labels", "_label", "_mask", "_masks", "-labels", "-label", "-mask", "-masks"]:
        aliases.add(stem.replace(suffix, ""))
    return {item for item in aliases if item}


def build_flat_stem_index(path: Path, valid_exts: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not path.exists():
        return index
    for file_path in sorted(path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in valid_exts:
            continue
        for key in stem_aliases(file_path.stem):
            index[key] = file_path
    return index


def build_stem_index(path: Path, valid_exts: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not path.exists():
        return index
    for file_path in path.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in valid_exts:
            continue
        for key in stem_aliases(file_path.stem):
            index[key] = file_path
    return index


def find_best_overlay_candidates(
    overlay_index: dict[str, dict[str, Path]], num_samples: int
) -> list[tuple[str, str, Path]]:
    """Find best overlay candidates that appear across most models."""
    image_ids: Counter = Counter()
    for model_overlays in overlay_index.values():
        image_ids.update(model_overlays.keys())

    candidates = []
    for image_id, count in image_ids.most_common():
        if count < 2:
            break
        paths = {}
        for model_name, model_overlays in overlay_index.items():
            if image_id in model_overlays:
                paths[model_name] = model_overlays[image_id]
        candidates.append((image_id, paths))
        if len(candidates) >= num_samples:
            break
    return candidates


def load_image(path: Path) -> np.ndarray | None:
    """Load image from path."""
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img


def draw_label_text(
    img: np.ndarray, text: str, font_scale: float = 0.6, thickness: int = 2
) -> np.ndarray:
    """Draw bold label text on image."""
    rows = img.shape[0]
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = max(8, (img.shape[1] - text_size[0]) // 2)
    text_y = max(text_size[1] + 8, rows - 8)
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness * 2, cv2.LINE_AA)
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def create_model_row(
    overlay_paths: dict[str, Path],
    model_name: str,
    display_name: str,
    sample_ids: list[str],
    size_tag: str,
) -> np.ndarray:
    """Create a row of overlay images for one model."""
    canvases = []
    for sample_id in sample_ids:
        if sample_id in overlay_paths:
            img = load_image(overlay_paths[sample_id])
        else:
            img = np.full((512, 512, 3), 128, dtype=np.uint8)
            cv2.putText(img, "N/A", (200, 256), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        img = draw_label_text(img, sample_id)
        canvases.append(img)

    label_canvas = np.full((512, 120, 3), 30, dtype=np.uint8)
    label_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    label_canvas = draw_label_text(label_canvas, f"{label_name}", font_scale=0.8, thickness=2)
    canvases.insert(0, label_canvas)

    row = np.concatenate(canvases, axis=1)
    return row


def create_prediction_matrix(
    overlay_index: dict[str, dict[str, Path]],
    models: list[str],
    sample_ids: list[str],
    size_tag: str,
) -> np.ndarray:
    """Create full prediction matrix for one dataset/size."""
    rows = []
    for model_name in models:
        model_overlays = overlay_index.get(model_name, {})
        row = create_model_row(model_overlays, model_name, MODEL_DISPLAY_NAMES.get(model_name, model_name), sample_ids, size_tag)
        rows.append(row)

    header_height = 60
    header = np.full((header_height, rows[0].shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(header, "Sample ID", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    sample_cols = rows[0].shape[1] // (len(sample_ids) + 1)
    for i, sid in enumerate(sample_ids):
        x = (i + 1) * sample_cols + 8
        cv2.putText(header, sid[:12], (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    rows.insert(0, header)

    return np.concatenate(rows, axis=0)


def save_matrix(
    matrix: np.ndarray,
    output_path: Path,
    size_tag: str,
    dataset_id: str,
) -> None:
    """Save matrix image and metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), matrix)
    meta_path = output_path.with_suffix(".json")
    json.dump(
        {"size_tag": size_tag, "dataset_id": dataset_id, "path": str(output_path)},
        meta_path.open("w", encoding="utf-8"),
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    ranked_models = rank_top_models(args.top_k_models)
    selected_models = [m["model_name"] for m in ranked_models]
    print(f"Selected models: {selected_models}")

    all_matrices = []
    for dataset_spec in DATASET_SPECS:
        dataset_id = dataset_spec["dataset_id"]
        if dataset_id not in args.datasets:
            continue
        display_name = dataset_spec["display_name"]
        print(f"\nProcessing dataset: {display_name}")

        for size_tag in args.sizes:
            if size_tag not in dataset_spec["size_to_dataset_key"]:
                continue
            dataset_key = dataset_spec["size_to_dataset_key"][size_tag]
            print(f"  Size: {size_tag}")

            overlay_index = discover_overlay_paths(size_tag, dataset_key)
            available_models = [m for m in selected_models if m in overlay_index]
            print(f"    Available models: {available_models}")

            candidates = find_best_overlay_candidates(overlay_index, args.samples_per_dataset)
            sample_ids = [c[0] for c in candidates]
            print(f"    Sample IDs: {sample_ids}")

            if not sample_ids:
                print(f"    No valid samples found, skipping...")
                continue

            matrix = create_prediction_matrix(overlay_index, available_models, sample_ids, size_tag)
            output_path = output_root / f"{dataset_id}_{size_tag}_matrix.png"
            save_matrix(matrix, output_path, size_tag, dataset_id)
            all_matrices.append({"dataset": display_name, "size": size_tag, "path": str(output_path)})

    meta_path = output_root / "matrices_metadata.json"
    json.dump(all_matrices, meta_path.open("w", encoding="utf-8"))
    print(f"\nSaved {len(all_matrices)} matrices to {output_root}")


if __name__ == "__main__":
    main()
