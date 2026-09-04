from __future__ import annotations

import copy
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class ModelEMA:
    """Exponential Moving Average of model weights for stable validation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_model.load_state_dict(state_dict)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int, scheduler_type: str = "cosine", step_size: int = 20):
    if scheduler_type == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def init_csv_logger(csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "dice", "iou", "precision", "recall", "f1", "bf_score", "hd95"]
        )


def append_csv_log(csv_path: str | Path, row: Dict[str, float | int]) -> None:
    with Path(csv_path).open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["epoch"],
                row["train_loss"],
                row["val_loss"],
                row["dice"],
                row["iou"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["bf_score"],
                row["hd95"],
            ]
        )


def metric_score(metrics: Dict[str, float], best_by: str) -> float:
    """Return a scalar score for checkpoint selection.

    Scores are normalized so larger is better. Metric names keep their
    historical behavior, while boundary_score favors cleaner contours.
    """
    if best_by == "hd95":
        return -float(metrics["hd95"])
    if best_by == "boundary_score":
        return (
            float(metrics["dice"])
            + float(metrics["bf_score"])
            - 0.02 * float(metrics["hd95"])
        )
    return float(metrics[best_by])


def is_better(current: float, best: float, best_by: str, min_delta: float = 0.0) -> bool:
    return current > (best + min_delta)


def initial_best(best_by: str) -> float:
    return -float("inf")


def build_early_stopping_state(cfg: Dict[str, Any], best_by: str) -> Dict[str, Any]:
    train_cfg = cfg["train"]
    patience = max(0, int(train_cfg.get("early_stopping_patience", 0)))
    min_delta = max(0.0, float(train_cfg.get("early_stopping_min_delta", 1.0e-4)))
    min_epochs = max(
        int(train_cfg.get("warmup_epochs", 0)),
        int(train_cfg.get("early_stopping_min_epochs", 0)),
    )
    metrics_interval = max(1, int(train_cfg.get("metrics_interval", 1)))
    state_path = Path(
        train_cfg.get(
            "early_stopping_state_path",
            Path(train_cfg["save_dir"]) / "early_stopping_state.json",
        )
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "enabled": patience > 0,
        "monitor": best_by,
        "patience": patience,
        "min_delta": min_delta,
        "min_epochs": min_epochs,
        "metrics_interval": metrics_interval,
        "state_path": str(state_path),
        "best_epoch": None,
        "best_score": None,
        "best_metrics": {},
        "last_epoch": 0,
        "last_score": None,
        "last_metrics": {},
        "last_improved": False,
        "evaluations_completed": 0,
        "evaluations_since_improvement": 0,
        "monitoring_started": False,
        "stopped_early": False,
        "stop_epoch": None,
        "stop_reason": "",
    }


def _floatify_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {str(key): float(value) for key, value in metrics.items()}


def update_early_stopping_state(
    state: Dict[str, Any],
    epoch: int,
    current_score: float | None = None,
    best_score: float | None = None,
    metrics: Dict[str, Any] | None = None,
    improved: bool = False,
) -> Dict[str, Any]:
    state["last_epoch"] = int(epoch)
    state["monitoring_started"] = epoch >= int(state["min_epochs"])
    if current_score is None:
        return state

    state["evaluations_completed"] += 1
    state["last_score"] = float(current_score)
    state["best_score"] = float(best_score) if best_score is not None else None
    state["last_improved"] = bool(improved)
    state["last_metrics"] = _floatify_metrics(metrics or {})

    if improved:
        state["best_epoch"] = int(epoch)
        state["evaluations_since_improvement"] = 0
        state["best_metrics"] = _floatify_metrics(metrics or {})
    elif state["monitoring_started"]:
        state["evaluations_since_improvement"] += 1

    should_stop = (
        state["enabled"]
        and state["monitoring_started"]
        and state["evaluations_since_improvement"] >= state["patience"]
    )
    state["stopped_early"] = bool(should_stop)
    state["stop_epoch"] = int(epoch) if should_stop else None
    if should_stop:
        state["stop_reason"] = (
            f"no improvement in '{state['monitor']}' for "
            f"{state['evaluations_since_improvement']} evaluations"
        )
    return state


def save_early_stopping_state(state: Dict[str, Any]) -> None:
    payload = dict(state)
    state_path = Path(payload.pop("state_path"))
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_early_stopping_status(state: Dict[str, Any]) -> str:
    if not state["enabled"]:
        return "[EarlyStopping] disabled"
    last_score = state["last_score"]
    best_score = state["best_score"]
    if last_score is None or best_score is None:
        return (
            f"[EarlyStopping] monitor={state['monitor']} patience={state['patience']} "
            f"min_delta={state['min_delta']:.1e} min_epochs={state['min_epochs']}"
        )
    return (
        f"[EarlyStopping] monitor={state['monitor']} current={last_score:.6f} "
        f"best={best_score:.6f} best_epoch={state['best_epoch']} "
        f"bad_evals={state['evaluations_since_improvement']}/{state['patience']} "
        f"min_delta={state['min_delta']:.1e} min_epochs={state['min_epochs']}"
    )
