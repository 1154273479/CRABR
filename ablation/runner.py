"""Ablation runner for CRABR paper-style experiments.

Runs 3-stage progressive ablation:
    Stage 1: Baseline
    Stage 2: + FE + HCLF + DualBranch
    Stage 3: + GeoLoop
    Stage 4: + ERGA (Full)

Supports multiple datasets, seeds, and generates summary statistics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from utils.config import save_json

from .groups import DEFAULT_CONFIGS, DEFAULT_SEEDS, GROUP_SPECS, DATASET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CRABR ablation experiments")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS,
                        help="Dataset configs to include")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds for repeated runs")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--quick", action="store_true",
                        help="Debug mode with 30 epochs and a single seed")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--eval-split", type=str, default="test",
                        choices=["val", "test"],
                        help="Split used for final reporting")
    parser.add_argument("--best-by", type=str, default="boundary_score",
                        help="Metric for checkpoint selection")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=2.0e-4,
                        help="Minimum improvement for early stopping")
    parser.add_argument("--min-epochs", type=int, default=80,
                        help="Do not early-stop before this epoch")
    parser.add_argument("--output-root", type=str, default="ablation_results/crabr_512_paper")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse existing test metrics if present")
    return parser.parse_args()


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep update base dict with override values."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def sanitize_name(name: str) -> str:
    """Sanitize experiment name for use in paths."""
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def load_raw_config(config_path: Path) -> Dict[str, Any]:
    """Load raw YAML config file."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config file: {config_path}")
    return cfg


def debug_event(hypothesis_id: str, location: str, msg: str,
                data: Dict[str, Any] | None = None, run_id: str = "pre") -> None:
    """Send debug event to local debug server if available."""
    env_path = Path(".dbg/quick-ablation-bus-error.env")
    url = "http://127.0.0.1:7777/event"
    session_id = "quick-ablation-bus-error"
    try:
        if env_path.exists():
            env_text = env_path.read_text(encoding="utf-8")
            for line in env_text.splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    url = line.split("=", 1)[1].strip() or url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
        payload = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data or {},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=1).read()
    except Exception:
        pass


def get_seg_classes(cfg: Dict[str, Any]) -> int:
    """Get number of segmentation classes from config."""
    task_mode = cfg["model"]["task_mode"]
    num_classes = int(cfg["model"]["num_classes"])
    return 1 if task_mode == "binary" else num_classes


def dataset_label(config_stem: str) -> str:
    """Get display label for dataset."""
    return DATASET_LABELS.get(config_stem, config_stem)


def iter_selected_specs(groups: Iterable[str]) -> List[Dict[str, Any]]:
    """Iterate over selected ablation specifications."""
    selected = []
    for group_name in groups:
        if group_name not in GROUP_SPECS:
            raise ValueError(f"Unknown group: {group_name}")
        for spec in GROUP_SPECS[group_name]:
            item = copy.deepcopy(spec)
            item["group"] = group_name
            selected.append(item)
    return selected


def is_experiment_applicable(cfg: Dict[str, Any], spec: Dict[str, Any]) -> tuple[bool, str]:
    """Check if experiment is applicable to current dataset."""
    if spec.get("requires_seg_classes_gt1", False) and get_seg_classes(cfg) <= 1:
        return False, "Dataset has seg_classes <= 1, not applicable"
    return True, ""


def build_variant_name(config_stem: str, group_name: str,
                      experiment_name: str, seed: int) -> str:
    """Build unique variant name for experiment."""
    return sanitize_name(f"{config_stem}__{group_name}__{experiment_name}__seed{seed}")


def configure_output_paths(cfg: Dict[str, Any], output_root: Path,
                          size_tag: str, variant_name: str) -> None:
    """Configure output paths for experiment artifacts."""
    artifact_root = output_root / "artifacts"
    train_cfg = cfg.setdefault("train", {})
    infer_cfg = cfg.setdefault("inference", {})

    save_dir = (artifact_root / "checkpoints" / size_tag / variant_name).resolve()
    log_dir = (artifact_root / "logs" / size_tag / variant_name).resolve()
    pred_dir = (artifact_root / "predictions" / size_tag / variant_name).resolve()
    result_json = (artifact_root / "results" / size_tag /
                   f"{variant_name}_metrics.json").resolve()

    train_cfg["save_dir"] = str(save_dir)
    train_cfg["log_dir"] = str(log_dir)
    train_cfg["csv_log_path"] = str((log_dir / "train_log.csv").resolve())
    train_cfg["best_metrics_path"] = str((save_dir / "best_metrics.json").resolve())
    train_cfg["best_model_path"] = str((save_dir / "best_model.pth").resolve())
    train_cfg["last_model_path"] = str((save_dir / "last_model.pth").resolve())
    train_cfg["early_stopping_state_path"] = str(
        (save_dir / "early_stopping_state.json").resolve())
    infer_cfg["prediction_dir"] = str(pred_dir)
    infer_cfg["result_json"] = str(result_json)


def normalize_generated_runtime(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Normalize runtime configuration for generated config."""
    train_cfg = cfg.setdefault("train", {})
    data_cfg = cfg.setdefault("data", {})

    train_cfg["epochs"] = 30 if args.quick else int(args.epochs)
    train_cfg["device"] = "cuda"
    train_cfg["best_by"] = args.best_by
    train_cfg["early_stopping_patience"] = int(args.patience)
    train_cfg["early_stopping_min_delta"] = float(args.min_delta)
    train_cfg["early_stopping_min_epochs"] = int(args.min_epochs)

    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if gpu_ids:
        train_cfg["gpu_ids"] = gpu_ids
    else:
        train_cfg["gpu_ids"] = [int(item) for item in train_cfg.get("gpu_ids", [])]
    train_cfg["multi_gpu"] = len(train_cfg["gpu_ids"]) > 1

    if args.quick:
        data_cfg["num_workers"] = 0
        data_cfg["pin_memory"] = False
        data_cfg["persistent_workers"] = False
    elif int(data_cfg.get("num_workers", 0)) <= 0:
        data_cfg["num_workers"] = 0
        data_cfg["pin_memory"] = False
        data_cfg["persistent_workers"] = False

    if int(data_cfg.get("num_workers", 0)) <= 0:
        data_cfg.pop("prefetch_factor", None)


def make_generated_config(
    base_config_path: Path,
    spec: Dict[str, Any],
    output_root: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Path, Dict[str, Any], str]:
    """Generate temporary config file for one experiment run."""
    raw_cfg = load_raw_config(base_config_path)
    cfg = copy.deepcopy(raw_cfg)
    config_stem = base_config_path.stem
    image_size = cfg.get("data", {}).get("image_size", [512, 512])
    if isinstance(image_size, list):
        size_tag = str(image_size[0])
    else:
        size_tag = str(image_size)
    variant_name = build_variant_name(config_stem, spec["group"],
                                      spec["name"], seed)

    normalize_generated_runtime(cfg, args)

    model_ablation = cfg.setdefault("model", {}).setdefault("ablation", {})
    deep_update(model_ablation, spec.get("model_overrides", {}))
    deep_update(cfg.setdefault("loss", {}), spec.get("loss_overrides", {}))
    final_gpu_ids = [int(item) for item in cfg.setdefault("train", {}).get("gpu_ids", [])]
    cfg["train"]["gpu_ids"] = final_gpu_ids
    cfg["train"]["multi_gpu"] = len(final_gpu_ids) > 1
    configure_output_paths(cfg, output_root, size_tag, variant_name)

    generated_dir = output_root / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_dir / f"_temp_kfold_{variant_name}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return config_path, cfg, variant_name


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run subprocess command."""
    return subprocess.run(cmd, cwd=str(cwd), text=True)


def run_training(config_path: Path, repo_root: Path,
                  gpu_ids: str) -> subprocess.CompletedProcess[str]:
    """Run training for one experiment."""
    cmd = [sys.executable, "train.py", "--config", str(config_path)]
    if gpu_ids.strip():
        cmd.extend(["--gpu-ids", gpu_ids.strip()])
    return run_command(cmd, repo_root)


def run_inference(config_path: Path, repo_root: Path,
                  split: str) -> subprocess.CompletedProcess[str]:
    """Run inference for one experiment."""
    cmd = [sys.executable, "infer.py", "--config", str(config_path), "--split", split]
    return run_command(cmd, repo_root)


def count_parameters(config_path: Path) -> tuple[int, int]:
    """Count model parameters."""
    debug_event("A", "runner.py:count_parameters:start",
                "[DEBUG] entering count_parameters", {"config_path": str(config_path)})
    debug_event("A", "runner.py:count_parameters:before_import_torch",
                "[DEBUG] before import torch")
    import torch
    debug_event("A", "runner.py:count_parameters:after_import_torch",
                "[DEBUG] after import torch")

    debug_event("A", "runner.py:count_parameters:before_import_model",
                "[DEBUG] before import ERGASegmenter")
    from models import ERGASegmenter
    from utils.config import load_config
    debug_event("A", "runner.py:count_parameters:after_import_model",
                "[DEBUG] after import ERGASegmenter")

    cfg = load_config(str(config_path))
    debug_event("A", "runner.py:count_parameters:after_load_config",
                "[DEBUG] after load_config")
    device = torch.device("cpu")
    debug_event("A", "runner.py:count_parameters:before_build_model",
                "[DEBUG] before build model")
    model = ERGASegmenter(cfg).to(device)
    debug_event("A", "runner.py:count_parameters:after_build_model",
                "[DEBUG] after build model")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    debug_event("A", "runner.py:count_parameters:done",
                "[DEBUG] leaving count_parameters",
                {"total_params": total_params, "trainable_params": trainable_params})
    return total_params, trainable_params


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def metric_or_none(metrics: Dict[str, Any], key: str) -> float | None:
    """Get metric value or None."""
    value = metrics.get(key)
    return None if value is None else float(value)


def aggregate_runs(run_records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Aggregate results across seeds."""
    grouped: Dict[tuple[str, str, str], list[Dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        grouped[(record["dataset"], record["group"],
                 record["experiment"])].append(record)

    aggregated = []
    for (dataset_name, group_name, experiment_name), items in sorted(grouped.items()):
        ok_items = [item for item in items if item["status"] == "ok"]
        template = items[0]
        row = {
            "dataset": dataset_name,
            "group": group_name,
            "experiment": experiment_name,
            "label": template["label"],
            "stage": template.get("stage", 0),
            "status": "ok" if ok_items else template["status"],
            "note": template.get("note", ""),
            "num_runs": len(items),
            "num_ok_runs": len(ok_items),
            "params": ok_items[0]["params"] if ok_items else None,
            "trainable_params": ok_items[0]["trainable_params"] if ok_items else None,
        }
        for metric_name in ("dice", "iou", "bf_score", "hd95", "recall", "precision", "f1"):
            values = [item[metric_name] for item in ok_items
                      if item.get(metric_name) is not None]
            row[f"{metric_name}_mean"] = (sum(values) / len(values)) if values else None
            if values:
                mean_value = row[f"{metric_name}_mean"]
                row[f"{metric_name}_std"] = (
                    (sum((value - mean_value) ** 2 for value in values) / len(values)) ** 0.5
                )
            else:
                row[f"{metric_name}_std"] = None
        aggregated.append(row)

    # Compute delta vs full model
    full_lookup = {
        (item["dataset"], item["group"]): item
        for item in aggregated
        if item["experiment"] == "full"
    }
    for item in aggregated:
        baseline = full_lookup.get((item["dataset"], item["group"]))
        if baseline and item["status"] == "ok":
            for metric_name in ("dice", "iou", "bf_score", "hd95", "recall"):
                current_value = item.get(f"{metric_name}_mean")
                base_value = baseline.get(f"{metric_name}_mean")
                item[f"delta_{metric_name}_vs_full"] = (
                    None if current_value is None or base_value is None
                    else current_value - base_value
                )
        else:
            for metric_name in ("dice", "iou", "bf_score", "hd95", "recall"):
                item[f"delta_{metric_name}_vs_full"] = None
    return aggregated


def write_csv(rows: list[Dict[str, Any]], csv_path: Path) -> None:
    """Write results to CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "group", "experiment", "label", "stage", "status",
        "num_runs", "num_ok_runs",
        "dice_mean", "dice_std", "iou_mean", "iou_std",
        "bf_score_mean", "bf_score_std", "hd95_mean", "hd95_std",
        "recall_mean", "recall_std", "precision_mean", "precision_std",
        "f1_mean", "f1_std",
        "params", "trainable_params",
        "delta_dice_vs_full", "delta_iou_vs_full",
        "delta_bf_score_vs_full", "delta_hd95_vs_full", "delta_recall_vs_full",
        "note",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(mean_value: float | None, std_value: float | None,
                  digits: int = 4) -> str:
    """Format metric with mean±std."""
    if mean_value is None:
        return "N/A"
    if std_value is None:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f}±{std_value:.{digits}f}"


def format_delta(value: float | None, digits: int = 4) -> str:
    """Format delta value."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}"


def write_markdown(rows: list[Dict[str, Any]], md_path: Path, eval_split: str) -> None:
    """Write results to Markdown file."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["group"])].append(row)

    lines = [
        "# CRABR 512尺寸论文版消融实验汇总",
        "",
        f"- 训练策略: `200 epochs` 上限 + early stopping + 单次 train/val/test 划分",
        f"- 报告指标: `{eval_split}` split 的 `Dice / IoU / BF Score / HD95 / Recall`",
        "- 统计方式: 多随机种子 `mean±std`",
        "",
    ]
    for (dataset_name, group_name), items in sorted(grouped.items()):
        # Sort by stage
        items_sorted = sorted(items, key=lambda x: x.get("stage", 0))
        lines.append(f"## {DATASET_LABELS.get(dataset_name, dataset_name)} | {group_name}")
        lines.append("")
        lines.append("| Stage | 版本 | 状态 | Dice | IoU | BF Score | HD95 | Recall | Params | ΔDice | ΔBF |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in items_sorted:
            lines.append(
                "| {stage} | {label} | {status} | {dice} | {iou} | {bf} | {hd95} | {recall} | {params} | {ddice} | {dbf} |".format(
                    stage=item.get("stage", ""),
                    label=item["label"],
                    status=item["status"],
                    dice=format_metric(item["dice_mean"], item["dice_std"]),
                    iou=format_metric(item["iou_mean"], item["iou_std"]),
                    bf=format_metric(item["bf_score_mean"], item["bf_score_std"]),
                    hd95=format_metric(item["hd95_mean"], item["hd95_std"], digits=3),
                    recall=format_metric(item["recall_mean"], item["recall_std"]),
                    params=item["params"] if item["params"] is not None else "N/A",
                    ddice=format_delta(item["delta_dice_vs_full"]),
                    dbf=format_delta(item["delta_bf_score_vs_full"]),
                )
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(rows: list[Dict[str, Any]]) -> None:
    """Print summary to console."""
    print("\n" + "=" * 120)
    print("CRABR Ablation Summary")
    print("=" * 120)
    print(f"{'Dataset':<12} {'Group':<16} {'Variant':<20} {'Stage':<5} {'Status':<12} "
          f"{'Dice':>10} {'BF':>10} {'HD95':>10}")
    print("-" * 120)
    for row in sorted(rows, key=lambda x: (x["dataset"], x.get("stage", 0))):
        print(
            f"{row['dataset']:<12} {row['group']:<16} {row['experiment']:<20} "
            f"{row.get('stage', ''):<5} {row['status']:<12} "
            f"{format_metric(row['dice_mean'], row['dice_std']):>10} "
            f"{format_metric(row['bf_score_mean'], row['bf_score_std']):>10} "
            f"{format_metric(row['hd95_mean'], row['hd95_std'], digits=3):>10}"
        )


def main() -> None:
    """Main entry point for ablation runner."""
    args = parse_args()
    if args.quick:
        args.epochs = 30
        if len(args.seeds) > 1:
            args.seeds = [args.seeds[0]]

    repo_root = Path(__file__).parent.parent.resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_specs = iter_selected_specs(["main_modules"])
    run_records: list[Dict[str, Any]] = []

    for config_arg in args.configs:
        base_config_path = (repo_root / config_arg).resolve()
        if not base_config_path.exists():
            raise FileNotFoundError(f"Config not found: {base_config_path}")
        raw_cfg = load_raw_config(base_config_path)
        config_stem = base_config_path.stem

        print("\n" + "=" * 120)
        print(f"Dataset: {dataset_label(config_stem)} | Config: {config_stem}")
        print("=" * 120)

        for spec in selected_specs:
            applicable, reason = is_experiment_applicable(raw_cfg, spec)
            if not applicable:
                print(f"[N/A] {config_stem} | {spec['group']} | {spec['name']} | {reason}")
                run_records.append({
                    "dataset": config_stem,
                    "group": spec["group"],
                    "experiment": spec["name"],
                    "label": spec["label"],
                    "stage": spec.get("stage", 0),
                    "seed": None,
                    "status": "not_applicable",
                    "note": reason,
                    "params": None,
                    "trainable_params": None,
                    "dice": None, "iou": None, "bf_score": None,
                    "hd95": None, "recall": None, "precision": None, "f1": None,
                })
                continue

            for seed in args.seeds:
                generated_config_path, generated_cfg, _ = make_generated_config(
                    base_config_path, spec, output_root, seed, args)
                result_json_path = Path(generated_cfg["inference"]["result_json"])
                best_metrics_path = Path(generated_cfg["train"]["best_metrics_path"])
                best_model_path = Path(generated_cfg["train"]["best_model_path"])

                print(f"[RUN] {config_stem} | {spec['name']} | seed={seed}")
                debug_event("A", "runner.py:main:before_count",
                            "[DEBUG] before count_parameters",
                            {"dataset": config_stem, "group": spec["group"],
                             "experiment": spec["name"], "seed": seed})
                params, trainable_params = count_parameters(generated_config_path)
                debug_event("B", "runner.py:main:before_train",
                            "[DEBUG] before train.py",
                            {"dataset": config_stem, "group": spec["group"],
                             "experiment": spec["name"], "seed": seed})

                if args.skip_existing and result_json_path.exists():
                    metrics_payload = load_json(result_json_path)
                    metrics = metrics_payload.get("overall", metrics_payload)
                    note = "loaded from existing test metrics"
                    status = "ok"
                else:
                    train_result = run_training(generated_config_path, repo_root,
                                               args.gpu_ids)
                    debug_event("B", "runner.py:main:after_train",
                                "[DEBUG] after train.py",
                                {"returncode": train_result.returncode,
                                 "best_model_exists": best_model_path.exists()})
                    if train_result.returncode != 0 or not best_model_path.exists():
                        print(f"[FAILED] train.py | {config_stem} | {spec['name']} | seed={seed}")
                        run_records.append({
                            "dataset": config_stem, "group": spec["group"],
                            "experiment": spec["name"], "label": spec["label"],
                            "stage": spec.get("stage", 0), "seed": seed,
                            "status": "failed", "note": "train.py failed",
                            "params": params, "trainable_params": trainable_params,
                            "dice": None, "iou": None, "bf_score": None,
                            "hd95": None, "recall": None, "precision": None, "f1": None,
                            "config_path": str(generated_config_path),
                            "best_metrics_path": str(best_metrics_path),
                            "result_json_path": str(result_json_path),
                        })
                        continue

                    infer_result = run_inference(generated_config_path, repo_root,
                                                 args.eval_split)
                    debug_event("C", "runner.py:main:after_infer",
                                "[DEBUG] after infer.py",
                                {"returncode": infer_result.returncode,
                                 "result_json_exists": result_json_path.exists()})
                    if infer_result.returncode != 0 or not result_json_path.exists():
                        print(f"[FAILED] infer.py | {config_stem} | {spec['name']} | seed={seed}")
                        run_records.append({
                            "dataset": config_stem, "group": spec["group"],
                            "experiment": spec["name"], "label": spec["label"],
                            "stage": spec.get("stage", 0), "seed": seed,
                            "status": "failed", "note": "infer.py failed",
                            "params": params, "trainable_params": trainable_params,
                            "dice": None, "iou": None, "bf_score": None,
                            "hd95": None, "recall": None, "precision": None, "f1": None,
                            "config_path": str(generated_config_path),
                            "best_metrics_path": str(best_metrics_path),
                            "result_json_path": str(result_json_path),
                        })
                        continue
                    metrics_payload = load_json(result_json_path)
                    metrics = metrics_payload.get("overall", metrics_payload)
                    note = ""
                    status = "ok"

                best_metrics = load_json(best_metrics_path) if best_metrics_path.exists() else {}
                run_records.append({
                    "dataset": config_stem, "group": spec["group"],
                    "experiment": spec["name"], "label": spec["label"],
                    "stage": spec.get("stage", 0), "seed": seed,
                    "status": status, "note": note,
                    "params": params, "trainable_params": trainable_params,
                    "dice": metric_or_none(metrics, "dice"),
                    "iou": metric_or_none(metrics, "iou"),
                    "bf_score": metric_or_none(metrics, "bf_score"),
                    "hd95": metric_or_none(metrics, "hd95"),
                    "recall": metric_or_none(metrics, "recall"),
                    "precision": metric_or_none(metrics, "precision"),
                    "f1": metric_or_none(metrics, "f1"),
                    "best_epoch": best_metrics.get("epoch"),
                    "config_path": str(generated_config_path),
                    "best_metrics_path": str(best_metrics_path),
                    "result_json_path": str(result_json_path),
                })
                print(f"[OK] {config_stem} | {spec['name']} | seed={seed} | "
                      f"Dice={metric_or_none(metrics, 'dice')}")

    aggregated_rows = aggregate_runs(run_records)
    summary_json_path = output_root / "summary.json"
    summary_csv_path = output_root / "ablation_summary_512.csv"
    summary_md_path = output_root / "ablation_summary_512.md"
    runs_json_path = output_root / "runs.json"

    save_json(
        summary_json_path,
        {
            "configs": args.configs,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "eval_split": args.eval_split,
            "best_by": args.best_by,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "min_epochs": args.min_epochs,
            "aggregated_rows": aggregated_rows,
        },
    )
    save_json(runs_json_path, {"run_records": run_records})
    write_csv(aggregated_rows, summary_csv_path)
    write_markdown(aggregated_rows, summary_md_path, args.eval_split)
    print_console_summary(aggregated_rows)

    print(f"\nSummary JSON: {summary_json_path}")
    print(f"Run Records: {runs_json_path}")
    print(f"CSV:   {summary_csv_path}")
    print(f"MD:    {summary_md_path}")


if __name__ == "__main__":
    main()
