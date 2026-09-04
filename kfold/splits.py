"""K-Fold split utilities for CRABR."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def collect_sample_ids(cfg: Dict) -> List[str]:
    """Collect all sample IDs from the dataset (train+val combined)."""
    data_cfg = cfg["data"]
    image_dirs = []

    if "image_dir" in data_cfg:
        image_dirs.append(Path(data_cfg["image_dir"]))
    else:
        for key in ("train_image_dir", "val_image_dir"):
            if key in data_cfg:
                image_dirs.append(Path(data_cfg[key]))

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ids = set()
    for d in image_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in exts:
                    ids.add(f.stem)
    return sorted(ids)


def compute_kfold_splits(
    sample_ids: List[str], n_folds: int, seed: int
) -> List[Dict[str, List[str]]]:
    """Generate K fold splits.

    Returns list of {"train": [...], "val": [...]} dicts.
    """
    import random

    ids = sorted(sample_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    fold_size = len(ids) // n_folds
    remainder = len(ids) % n_folds
    folds: List[List[str]] = []
    start = 0
    for i in range(n_folds):
        end = start + fold_size + (1 if i < remainder else 0)
        folds.append(ids[start:end])
        start = end

    splits = []
    for fold_idx in range(n_folds):
        val_ids = folds[fold_idx]
        train_ids = [sid for i, fold in enumerate(folds) if i != fold_idx for sid in fold]
        splits.append({"train": train_ids, "val": val_ids})
    return splits
