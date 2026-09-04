from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from metrics import evaluate_dataset
from models import ERGASegmenter
from utils.config import load_config
from utils.dataset import build_dataset_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split = args.split
    checkpoint = args.checkpoint or cfg["train"]["best_model_path"]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu")

    dataset = build_dataset_from_config(cfg, split=split, is_train=False)
    loader_kwargs = {
        "batch_size": cfg["train"]["val_batch_size"],
        "shuffle": False,
        "num_workers": int(cfg["data"].get("num_workers", 0)),
        "pin_memory": bool(cfg["data"].get("pin_memory", False)),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["data"].get("persistent_workers", False))
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    model = ERGASegmenter(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    if 'ema_model' in state:
        model.load_state_dict(state['ema_model'], strict=False)
    elif 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    metrics = evaluate_dataset(model, loader, cfg, device)
    print({key: round(value, 5) for key, value in metrics.items()})


if __name__ == "__main__":
    main()
