"""K-Fold CV runner for CRABR."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from utils.config import append_size_tag, get_image_size_tag, load_config

DELETE_CHECKPOINT_AFTER_INFER_MODELS = {"unet", "unetplusplus", "swin_transformer"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run K-Fold CV for CRABR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="Comma-separated GPU IDs, e.g. '0,1,2,3'")
    return parser.parse_args()


def get_dataset_name(config_path: str, cfg: Dict) -> str:
    return append_size_tag(Path(config_path).stem, get_image_size_tag(cfg))


def modify_cfg_for_fold(
    cfg: Dict,
    dataset_name: str,
    size_tag: str,
    fold_idx: int,
    split_json_path: str,
    gpu_ids: List[int] = None,
) -> Dict:
    """Deep-copy cfg and modify paths for this fold."""
    cfg = copy.deepcopy(cfg)
    fold_suffix = f"fold{fold_idx}"

    cfg["data"]["split_json"] = split_json_path

    if gpu_ids is not None:
        cfg["train"]["gpu_ids"] = gpu_ids
        cfg["train"]["multi_gpu"] = len(gpu_ids) > 1
        cfg["train"]["batch_size"] = len(gpu_ids)
        cfg["train"]["accumulate_steps"] = max(1, 8 // len(gpu_ids))
        cfg["train"]["val_batch_size"] = len(gpu_ids) * 2
        cfg["data"]["num_workers"] = len(gpu_ids)

    base_save = f"checkpoints/{size_tag}/{dataset_name}/{fold_suffix}"
    base_log = f"logs/{size_tag}/{dataset_name}/{fold_suffix}"
    cfg["train"]["save_dir"] = base_save
    cfg["train"]["log_dir"] = base_log
    cfg["train"]["csv_log_path"] = f"{base_log}/train_log.csv"
    cfg["train"]["best_metrics_path"] = f"{base_save}/best_metrics.json"
    cfg["train"]["best_model_path"] = f"{base_save}/best_model.pth"
    cfg["train"]["last_model_path"] = f"{base_save}/last_model.pth"

    cfg["inference"] = cfg.get("inference", {})
    cfg["inference"]["prediction_dir"] = f"predictions/{size_tag}/{dataset_name}/{fold_suffix}"
    cfg["inference"]["result_json"] = f"results/{size_tag}/{dataset_name}_{fold_suffix}_metrics.json"

    return cfg


def run_fold(
    fold_cfg: Dict,
    fold_idx: int,
    temp_config_path: Path,
) -> Dict:
    """Train and evaluate one fold. Returns metrics dict."""
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(fold_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"  Training Fold {fold_idx} [CRABR]")
    print(f"{'='*60}\n")

    train_cmd = [sys.executable, "train.py", "--config", str(temp_config_path)]
    result = subprocess.run(train_cmd, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        print(f"[FAILED] Training fold {fold_idx}")
        return {}

    best_model_path = Path(fold_cfg["train"]["best_model_path"])
    last_model_path = Path(fold_cfg["train"]["last_model_path"])

    print(f"\n  Inference Fold {fold_idx} [CRABR]\n")
    infer_cmd = [
        sys.executable,
        "infer.py",
        "--config",
        str(temp_config_path),
        "--checkpoint",
        fold_cfg["train"]["best_model_path"],
        "--split",
        "val",
    ]
    result = subprocess.run(infer_cmd, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        print(f"[FAILED] Inference fold {fold_idx}")

    metrics_path = Path(fold_cfg["train"]["best_metrics_path"])
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


def summarize_kfold_results(
    all_metrics: List[Dict],
    dataset_name: str,
    size_tag: str,
    n_folds: int,
) -> bool:
    """Print and save summary of K-fold results."""
    import numpy as np

    valid = [m for m in all_metrics if m]
    if not valid:
        print("[ERROR] No valid fold results to summarize.")
        return False

    keys = ["dice", "iou", "precision", "recall", "f1", "bf_score", "hd95"]
    print(f"\n{'='*60}")
    print(f"  {n_folds}-Fold CV Results: CRABR on {dataset_name}")
    print(f"{'='*60}")

    summary = {}
    for i, m in enumerate(all_metrics):
        if m:
            line = "  ".join(f"{k}={m.get(k, 0):.4f}" for k in keys if k in m)
            print(f"  Fold {i}: {line}")

    print(f"  {'─'*50}")
    for k in keys:
        values = [m[k] for m in valid if k in m]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            summary[k] = {"mean": float(mean), "std": float(std)}
            print(f"  {k:12s}: {mean:.4f} ± {std:.4f}")

    results_dir = Path("results") / size_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset_name}_kfold_summary.json"
    out_path.write_text(
        json.dumps({
            "model": "CRABR",
            "n_folds": n_folds,
            "per_fold": all_metrics,
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Saved to: {out_path}")
    return True


def run_kfold_experiment(args):
    """Run K-Fold CV experiment."""
    from .splits import collect_sample_ids, compute_kfold_splits

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return False

    cfg = load_config(args.config)

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids.strip() else None
    size_tag = get_image_size_tag(cfg)
    dataset_name = get_dataset_name(args.config, cfg)
    sample_ids = collect_sample_ids(cfg)
    print(f"Model: CRABR")
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {len(sample_ids)}")
    print(f"Folds: {args.folds}")

    if len(sample_ids) < args.folds:
        print(f"[ERROR] Not enough samples ({len(sample_ids)}) for {args.folds} folds")
        return False

    splits = compute_kfold_splits(sample_ids, args.folds, args.seed)

    splits_dir = Path("splits") / size_tag
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict] = []
    temp_config = Path(f"configs/_temp_kfold_{dataset_name}.yaml")

    for fold_idx in range(args.folds):
        split_json_path = str(splits_dir / f"{dataset_name}_fold{fold_idx}.json")
        Path(split_json_path).write_text(
            json.dumps(splits[fold_idx], indent=2), encoding="utf-8"
        )

        fold_cfg = modify_cfg_for_fold(cfg, dataset_name, size_tag, fold_idx,
                                        split_json_path, gpu_ids)
        metrics = run_fold(fold_cfg, fold_idx, temp_config)
        all_metrics.append(metrics)

    if temp_config.exists():
        temp_config.unlink()

    return summarize_kfold_results(all_metrics, dataset_name, size_tag, args.folds)


def main():
    args = parse_args()
    ok = run_kfold_experiment(args)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
