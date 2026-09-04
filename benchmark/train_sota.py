from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses import CompositeSegmentationLoss
from metrics import evaluate_dataset
from benchmark.models import MODEL_CHOICES, build_sota_model
from utils.config import append_size_tag, get_image_size_tag, load_config, save_json
from utils.dataset import build_dataset_from_config, build_train_val_datasets
from utils.training import (
    append_csv_log,
    build_early_stopping_state,
    build_scheduler,
    format_early_stopping_status,
    init_csv_logger,
    initial_best,
    is_better,
    metric_score,
    move_to_device,
    save_early_stopping_state,
    set_seed,
    update_early_stopping_state,
)


GENERIC_ZERO_LOSSES = (
    "lambda_aux",
    "lambda_aux_d2",
    "lambda_edge",
    "lambda_shared",
    "lambda_inner",
    "lambda_disjoint",
    "lambda_cons",
    "lambda_router",
    "lambda_boundary_dist",
    "lambda_sdm",
    "lambda_coarse_sdm",
    "lambda_curvature",
    "lambda_gate_entropy",
    "lambda_gate_diversity",
    "lambda_boundary_focal",
    "lambda_edge_consistency",
    "lambda_uncertainty",
    "lambda_unc_seg",
)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ERGA/SOTA comparison models on existing ERGA datasets")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--output-root", type=str, default="sota_runs")
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def parse_gpu_ids(gpu_ids: str) -> list[int]:
    values = []
    for item in gpu_ids.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def resolve_gpu_ids(cfg: Dict) -> list[int]:
    if cfg["train"].get("device", "cuda") != "cuda" or not torch.cuda.is_available():
        return []
    available = torch.cuda.device_count()
    requested = cfg["train"].get("gpu_ids", [])
    requested = [int(gpu_id) for gpu_id in requested] if requested else list(range(available))
    if not bool(cfg["train"].get("multi_gpu", False)):
        requested = requested[:1]
    invalid = [gpu_id for gpu_id in requested if gpu_id < 0 or gpu_id >= available]
    if invalid:
        raise ValueError(f"Invalid GPU ids {invalid}, available GPU ids: 0..{available - 1}")
    return requested


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def loader_kwargs(cfg: Dict, is_train: bool) -> Dict:
    data_cfg = cfg["data"]
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": cfg["train"]["batch_size"] if is_train else cfg["train"]["val_batch_size"],
        "shuffle": is_train,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return kwargs


def build_loaders(cfg: Dict):
    train_dataset, val_dataset = build_train_val_datasets(cfg)
    return (
        DataLoader(train_dataset, **loader_kwargs(cfg, is_train=True)),
        DataLoader(val_dataset, **loader_kwargs(cfg, is_train=False)),
    )


def build_test_loader(cfg: Dict) -> DataLoader | None:
    data_cfg = cfg["data"]
    split_json_path = data_cfg.get("split_json", "")
    has_explicit_test_split = bool(data_cfg.get("test_image_dir")) and bool(data_cfg.get("test_mask_dir"))
    has_saved_test_split = False
    if split_json_path and Path(split_json_path).exists():
        try:
            split_payload = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
            has_saved_test_split = bool(split_payload.get("test"))
        except Exception:
            has_saved_test_split = False

    if not has_explicit_test_split and not has_saved_test_split:
        return None

    try:
        test_dataset = build_dataset_from_config(cfg, split="test", is_train=False)
    except Exception as exc:
        print(f"[Test] Skip test evaluation because test dataset could not be built: {exc}")
        return None

    return DataLoader(test_dataset, **loader_kwargs(cfg, is_train=False))


def configure_loss_for_model(cfg: Dict, model_name: str) -> None:
    cfg["model"]["sota_model"] = model_name
    if model_name == "erga":
        return

    ablation = cfg["model"].setdefault("ablation", {})
    ablation.update(
        {
            "baseline": "plain_unet",
            "use_plain_unet_baseline": True,
            "disable_edge_branch": True,
            "disable_region_branch": True,
            "disable_overlap_prior": True,
            "disable_region_to_edge_constraint": True,
            "disable_prior_injection": True,
            "disable_erga": True,
            "disable_lightweight_attention": True,
            "disable_transformer": True,
        }
    )
    for key in GENERIC_ZERO_LOSSES:
        if key in cfg["loss"]:
            cfg["loss"][key] = 0.0


def configure_output_paths(cfg: Dict, config_path: str, model_name: str, experiment_name: str, output_root: str) -> Path:
    config_stem = Path(config_path).stem
    # `run_kfold.py` writes fold-specific paths into a temp config before invoking
    # SOTA train/infer. Preserve those explicit paths so each fold keeps its own
    # checkpoints, logs, predictions, and metrics instead of being overwritten.
    if config_stem.startswith("_temp_kfold_") and not experiment_name.strip():
        save_dir = Path(cfg["train"]["save_dir"])
        return save_dir.parent if save_dir.name else save_dir

    seed = int(cfg["train"]["seed"])
    run_name = experiment_name.strip() or f"{model_name}_seed{seed}"
    size_tag = get_image_size_tag(cfg)
    variant_name = append_size_tag(config_stem, size_tag)
    run_dir = Path(output_root) / size_tag / variant_name / run_name
    checkpoint_dir = run_dir / "checkpoints"

    cfg["train"]["save_dir"] = str(checkpoint_dir)
    cfg["train"]["log_dir"] = str(run_dir / "tensorboard")
    cfg["train"]["csv_log_path"] = str(run_dir / "train_log.csv")
    cfg["train"]["best_metrics_path"] = str(checkpoint_dir / "best_metrics.json")
    cfg["train"]["best_model_path"] = str(checkpoint_dir / "best_model.pth")
    cfg["train"]["last_model_path"] = str(checkpoint_dir / "last_model.pth")
    cfg["inference"]["prediction_dir"] = str(run_dir / "predictions")
    cfg["inference"]["result_json"] = str(run_dir / "results" / "test_metrics.json")
    return run_dir


def build_optimizer(model: torch.nn.Module, criterion: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    params = [{"params": model.parameters()}]
    extra = [param for param in criterion.parameters() if param.requires_grad]
    if extra:
        params.append({"params": extra, "lr": cfg["train"]["lr"]})
    return torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg) -> Dict[str, float]:
    model.train()
    meter = AverageMeter()
    accum_steps = int(cfg["train"].get("accumulate_steps", 1))
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        batch = move_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg["train"]["amp"])):
            preds = model(batch["image"])
            preds["logits"] = preds["seg_logits"]
            loss = criterion(preds, batch)["loss"] / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        meter.update(float(loss.detach()) * accum_steps, batch["image"].shape[0])
    return {"train_loss": meter.avg}


def estimate_val_loss(model, loader, criterion, device, cfg) -> float:
    model.eval()
    meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg["train"]["amp"])):
                preds = model(batch["image"])
                preds["logits"] = preds["seg_logits"]
                loss = criterion(preds, batch)["loss"]
            meter.update(float(loss.detach()), batch["image"].shape[0])
    return meter.avg


def checkpoint_payload(epoch: int, model, optimizer, scheduler, criterion, cfg, metrics: Dict[str, float], model_name: str) -> Dict:
    return {
        "epoch": epoch,
        "model_name": model_name,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "criterion": criterion.state_dict(),
        "config": cfg,
        "metrics": metrics,
    }


def load_early_stopping_state_if_available(state: Dict[str, Any]) -> Dict[str, Any]:
    state_path = Path(state["state_path"])
    if not state_path.exists():
        return state
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Resume] Failed to read early stopping state from {state_path}: {exc}")
        return state
    state.update(payload)
    state["state_path"] = str(state_path)
    return state


def load_best_value(cfg: Dict, best_by: str, fallback_metrics: Dict[str, float] | None = None) -> float:
    metrics_path = Path(cfg["train"]["best_metrics_path"])
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "best_score" in payload:
                return float(payload["best_score"])
        except Exception as exc:
            print(f"[Resume] Failed to read best metrics from {metrics_path}: {exc}")
    if fallback_metrics:
        return metric_score(fallback_metrics, best_by)
    return initial_best(best_by)


def load_training_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=device)
    if "model" not in state:
        raise KeyError(f"Checkpoint missing 'model' state: {checkpoint_path}")
    unwrap_model(model).load_state_dict(state["model"], strict=False)
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "criterion" in state:
        criterion.load_state_dict(state["criterion"])
    return state


def evaluate_checkpoint_on_test(
    cfg: Dict,
    checkpoint_path: str | Path,
    device: torch.device,
    model_name: str,
    split: str = "test",
) -> Dict[str, float] | None:
    loader = build_test_loader(cfg)
    if loader is None:
        print("[Test] No test split configured, skip test evaluation.")
        return None

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[Test] Checkpoint not found, skip test evaluation: {checkpoint_path}")
        return None

    model = build_sota_model(model_name, cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    metrics = evaluate_dataset(model, loader, cfg, device)
    payload = {
        "split": split,
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    save_json(cfg["inference"]["result_json"], payload)
    print(f"[Test] Saved {split} metrics to {cfg['inference']['result_json']}")
    print({key: round(value, 5) for key, value in metrics.items()})
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    configure_loss_for_model(cfg, args.model)
    run_dir = configure_output_paths(cfg, args.config, args.model, args.experiment_name, args.output_root)
    if args.gpu_ids.strip():
        cfg["train"]["gpu_ids"] = parse_gpu_ids(args.gpu_ids)

    set_seed(int(cfg["train"]["seed"]))
    gpu_ids = resolve_gpu_ids(cfg)
    if gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    Path(cfg["train"]["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["train"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["inference"]["result_json"]).parent.mkdir(parents=True, exist_ok=True)
    init_csv_logger(cfg["train"]["csv_log_path"])
    last_checkpoint_path = Path(cfg["train"]["last_model_path"])
    best_checkpoint_path = Path(cfg["train"]["best_model_path"])
    metric_keys = ["dice", "iou", "jaccard", "precision", "recall", "f1", "bf_score", "hd95"]
    best_by = cfg["train"].get("best_by", "dice")
    metrics_interval = max(1, int(cfg["train"].get("metrics_interval", 1)))
    last_val_metrics = {key: 0.0 for key in metric_keys}

    if not last_checkpoint_path.exists() and best_checkpoint_path.exists():
        print(f"[Startup] Found best checkpoint without resume checkpoint: {best_checkpoint_path}")
        evaluate_checkpoint_on_test(cfg, best_checkpoint_path, device, args.model, split="test")
        return

    train_loader, val_loader = build_loaders(cfg)
    model = build_sota_model(args.model, cfg).to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    elif len(gpu_ids) == 1:
        print(f"Using single GPU: {gpu_ids[0]}")
    else:
        print("Using CPU for training")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SOTA model: {args.model}")
    print(f"Run directory: {run_dir}")
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    criterion = CompositeSegmentationLoss(cfg).to(device)
    if criterion.pos_weight is None and criterion.class_weight is None:
        criterion.compute_pos_weight_from_dataset(train_loader.dataset)
        criterion = criterion.to(device)

    optimizer = build_optimizer(model, criterion, cfg)
    scheduler = build_scheduler(
        optimizer,
        cfg["train"]["epochs"],
        cfg["train"]["warmup_epochs"],
        cfg["train"].get("scheduler_type", "cosine"),
        cfg["train"].get("step_size", 20),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg["train"]["amp"]))
    writer = SummaryWriter(log_dir=cfg["train"]["log_dir"])

    early_stopping = build_early_stopping_state(cfg, best_by)
    start_epoch = 1
    best_value = initial_best(best_by)

    if last_checkpoint_path.exists():
        print(f"[Resume] Loading checkpoint from {last_checkpoint_path}")
        resume_state = load_training_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )
        last_val_metrics = {
            **last_val_metrics,
            **{key: float(value) for key, value in resume_state.get("metrics", {}).items() if key in last_val_metrics},
        }
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_value = load_best_value(cfg, best_by, fallback_metrics=resume_state.get("metrics", {}))
        early_stopping = load_early_stopping_state_if_available(early_stopping)
        print(f"[Resume] Resuming from epoch {start_epoch}")
    else:
        save_early_stopping_state(early_stopping)

    if start_epoch > cfg["train"]["epochs"]:
        print("[Resume] Checkpoint already reached configured epochs, skip training.")
        writer.close()
        eval_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
        evaluate_checkpoint_on_test(cfg, eval_checkpoint_path, device, args.model, split="test")
        return

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        train_log = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg)
        val_loss = estimate_val_loss(model, val_loader, criterion, device, cfg)
        run_full_metrics = (epoch % metrics_interval == 0) or (epoch == cfg["train"]["epochs"])
        val_metrics = evaluate_dataset(model, val_loader, cfg, device) if run_full_metrics else last_val_metrics
        if run_full_metrics:
            last_val_metrics = val_metrics
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_log["train_loss"], "val_loss": val_loss, **val_metrics}
        append_csv_log(cfg["train"]["csv_log_path"], row)
        writer.add_scalar("loss/train", train_log["train_loss"], epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"metric/{key}", value, epoch)

        if run_full_metrics:
            print(
                f"[Epoch {epoch:03d}/{cfg['train']['epochs']}] loss={train_log['train_loss']:.4f} "
                f"val_loss={val_loss:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
                f"bf={val_metrics['bf_score']:.4f} hd95={val_metrics['hd95']:.4f}"
            )
        else:
            print(f"[Epoch {epoch:03d}/{cfg['train']['epochs']}] loss={train_log['train_loss']:.4f} val_loss={val_loss:.4f}")

        payload = checkpoint_payload(epoch, model, optimizer, scheduler, criterion, cfg, val_metrics, args.model)
        if cfg["train"].get("save_last_model", True) and cfg["train"].get("last_model_path"):
            torch.save(payload, cfg["train"]["last_model_path"])
        if run_full_metrics:
            current_value = metric_score(val_metrics, best_by)
            improved = is_better(current_value, best_value, best_by, early_stopping["min_delta"])
            if improved:
                best_value = current_value
                torch.save(payload, cfg["train"]["best_model_path"])
                save_json(
                    cfg["train"]["best_metrics_path"],
                    {
                        "epoch": epoch,
                        "model_name": args.model,
                        "dataset_config": args.config,
                        "seed": int(cfg["train"]["seed"]),
                        "best_by": best_by,
                        "best_score": float(best_value),
                        **{key: float(value) for key, value in val_metrics.items()},
                    },
                )
            update_early_stopping_state(
                early_stopping,
                epoch=epoch,
                current_score=current_value,
                best_score=best_value,
                metrics=val_metrics,
                improved=improved,
            )
            save_early_stopping_state(early_stopping)
            print(format_early_stopping_status(early_stopping))

            if early_stopping["stopped_early"]:
                print(
                    f"Early stopping at epoch {epoch} after "
                    f"{early_stopping['evaluations_since_improvement']} metric evaluations "
                    f"without a {best_by} improvement larger than {early_stopping['min_delta']:.1e}."
                )
                break
        else:
            update_early_stopping_state(early_stopping, epoch=epoch)
            save_early_stopping_state(early_stopping)

    writer.close()
    eval_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
    evaluate_checkpoint_on_test(cfg, eval_checkpoint_path, device, args.model, split="test")


if __name__ == "__main__":
    main()
