from __future__ import annotations

import argparse
import copy
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
from models import ERGASegmenter
from utils.config import load_config, save_json
from utils.dataset import build_dataset_from_config, build_train_val_datasets
from utils.training import (
    AverageMeter,
    ModelEMA,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/jsrt_scr.yaml")
    parser.add_argument("--gpu-ids", type=str, default="")
    return parser.parse_args()


def parse_gpu_ids(gpu_ids: str) -> list[int]:
    values = []
    for item in gpu_ids.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def normalize_runtime_train_cfg(cfg: Dict, cli_gpu_ids: str) -> None:
    train_cfg = cfg.setdefault("train", {})
    train_cfg["epochs"] = int(train_cfg.get("epochs", 200))
    train_cfg["device"] = train_cfg.get("device", "cuda")
    train_cfg["gpu_ids"] = [int(gpu_id) for gpu_id in train_cfg.get("gpu_ids", [])]
    train_cfg["save_last_model"] = bool(train_cfg.get("save_last_model", True))

    if cli_gpu_ids.strip():
        train_cfg["gpu_ids"] = parse_gpu_ids(cli_gpu_ids)

    train_cfg["multi_gpu"] = len(train_cfg["gpu_ids"]) > 1
    train_cfg["early_stopping_patience"] = int(train_cfg.get("early_stopping_patience", 25))
    train_cfg["early_stopping_min_delta"] = float(train_cfg.get("early_stopping_min_delta", 2.0e-4))
    train_cfg["early_stopping_min_epochs"] = int(train_cfg.get("early_stopping_min_epochs", 80))
    train_cfg["metrics_interval"] = max(1, int(train_cfg.get("metrics_interval", 1)))


def resolve_gpu_ids(cfg: Dict) -> list[int]:
    if cfg["train"].get("device", "cuda") != "cuda" or not torch.cuda.is_available():
        return []
    available = torch.cuda.device_count()
    requested = cfg["train"].get("gpu_ids", [])
    requested = [int(gpu_id) for gpu_id in requested] if requested else []
    use_multi_gpu = bool(cfg["train"].get("multi_gpu", True))
    if not requested:
        requested = list(range(available))
    invalid = [gpu_id for gpu_id in requested if gpu_id < 0 or gpu_id >= available]
    if invalid:
        raise ValueError(f"Invalid GPU ids {invalid}, available GPU ids: 0..{available - 1}")
    if not use_multi_gpu and requested:
        return [requested[0]]
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
    train_loader = DataLoader(train_dataset, **loader_kwargs(cfg, is_train=True))
    val_loader = DataLoader(val_dataset, **loader_kwargs(cfg, is_train=False))
    return train_loader, val_loader


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


def build_optimizer(model: torch.nn.Module, criterion: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    params = [{"params": model.parameters()}]
    extra = [param for param in criterion.parameters() if param.requires_grad]
    if extra:
        params.append({"params": extra, "lr": cfg["train"]["lr"]})
    return torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg, ema=None) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    accum_steps = int(cfg["train"].get("accumulate_steps", 1))
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        batch = move_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg["train"]["amp"])):
            preds = model(batch["image"])
            if "seg_logits" in preds:
                preds["logits"] = preds["seg_logits"]
            for k, v in preds.items():
                if isinstance(v, torch.Tensor) and v.dim() == 1 and v.shape[0] == batch["image"].shape[0]:
                    preds[k] = v.mean()
            loss_dict = criterion(preds, batch)
            loss = loss_dict["loss"]
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(unwrap_model(model))
        loss_meter.update(float(loss.detach()) * accum_steps, batch["image"].shape[0])
    return {"train_loss": loss_meter.avg}


def estimate_val_loss(model, loader, criterion, device, cfg) -> float:
    model.eval()
    meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg["train"]["amp"])):
                preds = model(batch["image"])
                if "seg_logits" in preds:
                    preds["logits"] = preds["seg_logits"]
                for k, v in preds.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 1 and v.shape[0] == batch["image"].shape[0]:
                        preds[k] = v.mean()
                loss = criterion(preds, batch)["loss"]
                if loss.dim() > 0:
                    loss = loss.mean()
            meter.update(float(loss.detach()), batch["image"].shape[0])
    return meter.avg


def checkpoint_payload(epoch, model, optimizer, scheduler, criterion, cfg, metrics, ema=None):
    payload = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "criterion": criterion.state_dict(),
        "config": cfg,
        "metrics": metrics,
    }
    if ema is not None:
        payload["ema_model"] = ema.state_dict()
    return payload


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
    ema: ModelEMA | None,
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
    if ema is not None and "ema_model" in state:
        ema.load_state_dict(state["ema_model"])
    return state


def evaluate_checkpoint_on_test(
    cfg: Dict,
    checkpoint_path: str | Path,
    device: torch.device,
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

    model = ERGASegmenter(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "ema_model" in state:
        model.load_state_dict(state["ema_model"], strict=False)
    elif "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    metrics = evaluate_dataset(model, loader, cfg, device)
    payload = {
        "split": split,
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
    normalize_runtime_train_cfg(cfg, args.gpu_ids)
    set_seed(cfg["train"]["seed"])
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
    best_by = cfg["train"].get("best_by", "dice")
    metrics_interval = max(1, int(cfg["train"].get("metrics_interval", 1)))
    last_val_metrics = {"dice": 0.0, "iou": 0.0, "jaccard": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "bf_score": 0.0, "hd95": 0.0}

    if not last_checkpoint_path.exists() and best_checkpoint_path.exists():
        print(f"[Startup] Found best checkpoint without resume checkpoint: {best_checkpoint_path}")
        evaluate_checkpoint_on_test(cfg, best_checkpoint_path, device, split="test")
        return

    train_loader, val_loader = build_loaders(cfg)
    model = ERGASegmenter(cfg).to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    elif len(gpu_ids) == 1:
        print(f"Using single GPU: {gpu_ids[0]}")
    else:
        print("Using CPU for training")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    criterion = CompositeSegmentationLoss(cfg).to(device)
    if criterion.pos_weight is None and criterion.class_weight is None:
        criterion.compute_pos_weight_from_dataset(train_loader.dataset)
        criterion = criterion.to(device)
    optimizer = build_optimizer(model, criterion, cfg)
    scheduler = build_scheduler(optimizer, cfg["train"]["epochs"], cfg["train"]["warmup_epochs"], cfg["train"].get("scheduler_type", "cosine"), cfg["train"].get("step_size", 20))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg["train"]["amp"]))
    writer = SummaryWriter(log_dir=cfg["train"]["log_dir"])

    ema = ModelEMA(unwrap_model(model), decay=0.998)
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
            ema=ema,
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

    disable_curriculum_gradient = bool(cfg["model"].get("ablation", {}).get("disable_curriculum_gradient", False))

    if start_epoch > cfg["train"]["epochs"]:
        print("[Resume] Checkpoint already reached configured epochs, skip training.")
        writer.close()
        eval_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
        evaluate_checkpoint_on_test(cfg, eval_checkpoint_path, device, split="test")
        return

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        # Curriculum: gradually increase prior gradient coupling
        if disable_curriculum_gradient:
            ratio = 1.0
        elif epoch <= 20:
            ratio = 0.05
        elif epoch <= 50:
            ratio = 0.2
        else:
            ratio = 1.0
        unwrap_model(model).set_prior_grad_ratio(ratio)

        train_log = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg, ema=ema)
        val_loss = estimate_val_loss(ema.ema_model, val_loader, criterion, device, cfg)
        run_full_metrics = (epoch % metrics_interval == 0) or (epoch == cfg["train"]["epochs"])
        if run_full_metrics:
            val_metrics = evaluate_dataset(ema.ema_model, val_loader, cfg, device)
            last_val_metrics = val_metrics
        else:
            val_metrics = last_val_metrics
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_log["train_loss"], "val_loss": val_loss, **{k: val_metrics[k] for k in last_val_metrics}}
        append_csv_log(cfg["train"]["csv_log_path"], row)
        writer.add_scalar("loss/train", train_log["train_loss"], epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"metric/{key}", value, epoch)

        if run_full_metrics:
            print(f"[Epoch {epoch:03d}/{cfg['train']['epochs']}] loss={train_log['train_loss']:.4f} val_loss={val_loss:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} bf={val_metrics['bf_score']:.4f} hd95={val_metrics['hd95']:.4f}")
        else:
            print(f"[Epoch {epoch:03d}/{cfg['train']['epochs']}] loss={train_log['train_loss']:.4f} val_loss={val_loss:.4f}")

        payload = checkpoint_payload(epoch, model, optimizer, scheduler, criterion, cfg, val_metrics, ema=ema)
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
                    {"epoch": epoch, "best_by": best_by, "best_score": float(best_value), **{k: float(v) for k, v in val_metrics.items()}},
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
    evaluate_checkpoint_on_test(cfg, eval_checkpoint_path, device, split="test")


if __name__ == "__main__":
    main()
