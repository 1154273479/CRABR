from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import evaluate_dataset, logits_to_prediction, metrics_for_batch_per_class
from benchmark.models import MODEL_CHOICES, build_sota_model
from benchmark.train_sota import configure_loss_for_model, configure_output_paths
from utils.config import load_config, save_json
from utils.dataset import build_dataset_from_config
from utils.visualization import build_comparison_panel, build_overlay, palette_from_config, save_overlay, save_panel, save_prediction_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer/evaluate ERGA/SOTA comparison models")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--output-root", type=str, default="sota_runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save-images", action="store_true")
    return parser.parse_args()


def loader_kwargs(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": cfg["train"]["val_batch_size"],
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return kwargs


def save_sample_outputs(prediction_dir: Path, image_id: str, raw_image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, cfg: dict) -> None:
    task_mode = cfg["model"]["task_mode"]
    num_classes = int(cfg["model"]["num_classes"])
    class_names = cfg["data"].get("class_names", [])
    palette = palette_from_config(cfg)
    save_color_mask = cfg["inference"].get("save_color_mask", True)
    mode = {"binary": "exclusive", "multiclass": "exclusive", "multilabel": "independent"}.get(task_mode, task_mode)

    save_prediction_mask(prediction_dir / f"{image_id}.png", pred_mask, task_mode, palette=palette, colorize=save_color_mask)
    if cfg["inference"].get("save_per_class", False) and num_classes > 1:
        sample_dir = prediction_dir / image_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for class_idx in range(num_classes):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            if mode == "exclusive":
                class_mask = (pred_mask == class_idx).astype("uint8") if pred_mask.ndim == 2 else pred_mask[class_idx]
            else:
                class_mask = pred_mask[class_idx]
            save_prediction_mask(
                sample_dir / f"{class_name}.png",
                class_mask,
                "binary",
                palette=palette,
                colorize=save_color_mask,
                class_index=class_idx,
            )

    if cfg["inference"].get("save_overlay", True):
        overlay = build_overlay(raw_image, pred_mask, cfg["inference"]["overlay_alpha"], task_mode, palette)
        save_overlay(prediction_dir / f"{image_id}_overlay.png", overlay)
    if cfg["inference"].get("save_panel", True):
        panel = build_comparison_panel(
            raw_image,
            pred_mask,
            task_mode,
            cfg["inference"]["overlay_alpha"],
            ground_truth=gt_mask,
            palette=palette,
        )
        save_panel(prediction_dir / f"{image_id}_panel.png", panel)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    configure_loss_for_model(cfg, args.model)
    run_dir = configure_output_paths(cfg, args.config, args.model, args.experiment_name, args.output_root)
    checkpoint = Path(args.checkpoint or cfg["train"]["best_model_path"])
    if not checkpoint.exists():
        fallback = Path(cfg["train"]["last_model_path"])
        if fallback.exists():
            print(f"[WARN] best_model not found, falling back to: {fallback}")
            checkpoint = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    prediction_dir = Path(cfg["inference"]["prediction_dir"])
    prediction_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["inference"]["result_json"]).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu")
    dataset = build_dataset_from_config(cfg, split=args.split, is_train=False, include_raw_image=True)
    loader = DataLoader(dataset, **loader_kwargs(cfg))

    model = build_sota_model(args.model, cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("ema_model") or state.get("model") or state
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SOTA model: {args.model}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Run directory: {run_dir}")
    print(f"Model parameters: {total_params:,}")

    task_mode = cfg["model"]["task_mode"]
    num_classes = int(cfg["model"]["num_classes"])
    class_names = cfg["data"].get("class_names", [])
    ignore_bg = cfg["data"].get("ignore_background", True)
    normalized_mode = {"binary": "exclusive", "multiclass": "exclusive", "multilabel": "independent"}.get(task_mode, task_mode)
    start_idx = 1 if (ignore_bg and num_classes > 1 and normalized_mode == "exclusive") else 0
    effective_classes = num_classes - start_idx
    metric_keys = ["dice", "iou", "jaccard", "precision", "recall", "f1", "bf_score", "hd95"]
    per_class_sums: dict[int, dict[str, float]] = {}
    per_class_counts: dict[int, int] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg["train"]["amp"])):
                logits = model(image)["seg_logits"]
            pred = logits_to_prediction(logits, task_mode, cfg["data"]["threshold"]).cpu().numpy()
            gt_masks = batch["mask"].cpu().numpy()
            image_ids = batch["image_id"]
            raw_images = batch["raw_image"].numpy()

            batch_pc = metrics_for_batch_per_class(pred, gt_masks, task_mode, num_classes, ignore_background=ignore_bg)
            for sample_classes in batch_pc:
                for class_offset, class_metrics in enumerate(sample_classes):
                    if class_offset not in per_class_sums:
                        per_class_sums[class_offset] = {key: 0.0 for key in metric_keys}
                        per_class_counts[class_offset] = 0
                    for key in metric_keys:
                        per_class_sums[class_offset][key] += class_metrics[key]
                    per_class_counts[class_offset] += 1

            if not args.no_save_images:
                for idx, image_id in enumerate(image_ids):
                    save_sample_outputs(prediction_dir, image_id, raw_images[idx], pred[idx], gt_masks[idx], cfg)

    metrics = evaluate_dataset(model, DataLoader(dataset, **loader_kwargs(cfg)), cfg, device)
    per_class_result = {}
    for class_offset in range(effective_classes):
        class_idx = class_offset + start_idx
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        count = max(per_class_counts.get(class_offset, 0), 1)
        sums = per_class_sums.get(class_offset, {key: 0.0 for key in metric_keys})
        per_class_result[class_name] = {key: float(sums[key] / count) for key in metric_keys}

    result: dict[str, object] = {
        "model_name": args.model,
        "dataset_config": args.config,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "seed": int(cfg["train"]["seed"]),
        "overall": metrics,
        "per_class": per_class_result,
    }
    save_json(cfg["inference"]["result_json"], result)

    print("\n=== Overall ===")
    print({key: round(value, 5) for key, value in metrics.items()})
    print("\n=== Per-Class ===")
    for name, item in per_class_result.items():
        print(f"  {name}: dice={item['dice']:.4f} iou={item['iou']:.4f} bf={item['bf_score']:.4f} hd95={item['hd95']:.2f}")
    print(f"\nSaved results to: {cfg['inference']['result_json']}")


if __name__ == "__main__":
    main()
