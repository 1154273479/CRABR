from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMG_EXTS | {".npy", ".npz"}


def _list_files(path: Path, valid_exts: Sequence[str]) -> List[Path]:
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts])


def _stem_aliases(stem: str) -> set[str]:
    aliases = {stem}
    for suffix in ["_labels", "_label", "_mask", "_masks", "-labels", "-label", "-mask", "-masks"]:
        aliases.add(stem.replace(suffix, ""))
    return {a for a in aliases if a}


def _build_stem_index(path: Path, valid_exts: Sequence[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for file_path in _list_files(path, valid_exts):
        for key in _stem_aliases(file_path.stem):
            index[key] = file_path
    return index


def _build_flat_stem_index(path: Path, valid_exts: Sequence[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not path.exists():
        return index
    for file_path in sorted(path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in valid_exts:
            continue
        for key in _stem_aliases(file_path.stem):
            index[key] = file_path
    return index


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        return data[list(data.keys())[0]]
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if mask.ndim == 3 and mask.shape[-1] >= 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return mask


def _ensure_hwc(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return mask[..., None]
    if mask.ndim == 3 and mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2]:
        return np.transpose(mask, (1, 2, 0))
    return mask


def _resize_image(image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    h, w = image_size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    h, w = image_size
    if mask.ndim == 2:
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    channels = [cv2.resize(mask[..., idx], (w, h), interpolation=cv2.INTER_NEAREST) for idx in range(mask.shape[-1])]
    return np.stack(channels, axis=-1)


def _remap_mask(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    for c in range(mask.shape[0]):
        mask[c] = cv2.remap(mask[c], map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return mask


def _warp_mask(mask: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    for c in range(mask.shape[0]):
        mask[c] = cv2.warpAffine(mask[c], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return mask


def _elastic_deform(image: np.ndarray, mask: np.ndarray, alpha: float = 80.0, sigma: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = _remap_mask(mask, map_x, map_y)
    return image, mask


def _affine_transform(image: np.ndarray, mask: np.ndarray, rotate_limit: float = 15.0, scale_range: tuple = (0.9, 1.1)) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    angle = random.uniform(-rotate_limit, rotate_limit)
    scale = random.uniform(*scale_range)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = _warp_mask(mask, M, w, h)
    return image, mask


def _grid_distortion(image: np.ndarray, mask: np.ndarray, num_steps: int = 5, distort_limit: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    x_steps = [1.0 + random.uniform(-distort_limit, distort_limit) for _ in range(num_steps + 1)]
    y_steps = [1.0 + random.uniform(-distort_limit, distort_limit) for _ in range(num_steps + 1)]

    x_grid = np.linspace(0, w, num_steps + 1)
    y_grid = np.linspace(0, h, num_steps + 1)

    xx = np.zeros(w, dtype=np.float32)
    prev = 0.0
    for i in range(num_steps):
        start_x = int(x_grid[i])
        end_x = int(x_grid[i + 1])
        if end_x <= start_x:
            continue
        segment_len = (end_x - start_x) * x_steps[i]
        xx[start_x:end_x] = np.linspace(prev, prev + segment_len, end_x - start_x, dtype=np.float32)
        prev += segment_len
    xx = xx * (w - 1) / max(prev, 1e-6)

    yy = np.zeros(h, dtype=np.float32)
    prev = 0.0
    for i in range(num_steps):
        start_y = int(y_grid[i])
        end_y = int(y_grid[i + 1])
        if end_y <= start_y:
            continue
        segment_len = (end_y - start_y) * y_steps[i]
        yy[start_y:end_y] = np.linspace(prev, prev + segment_len, end_y - start_y, dtype=np.float32)
        prev += segment_len
    yy = yy * (h - 1) / max(prev, 1e-6)

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = _remap_mask(mask, map_x, map_y)
    return image, mask


def _apply_spatial_aug(image: np.ndarray, mask: np.ndarray, aug_cfg: Dict, is_train: bool) -> tuple[np.ndarray, np.ndarray]:
    if not is_train:
        return image, mask
    if random.random() < aug_cfg.get("elastic_p", 0.0):
        alpha = float(aug_cfg.get("elastic_alpha", 80.0))
        sigma = float(aug_cfg.get("elastic_sigma", 10.0))
        image, mask = _elastic_deform(image, mask, alpha=alpha, sigma=sigma)
    if random.random() < aug_cfg.get("affine_p", 0.0):
        rotate = float(aug_cfg.get("affine_rotate_limit", 15.0))
        scale = tuple(aug_cfg.get("affine_scale_range", [0.9, 1.1]))
        image, mask = _affine_transform(image, mask, rotate_limit=rotate, scale_range=scale)
    if random.random() < aug_cfg.get("grid_distort_p", 0.0):
        distort = float(aug_cfg.get("grid_distort_limit", 0.3))
        image, mask = _grid_distortion(image, mask, distort_limit=distort)
    return image, mask


def _apply_aug(image: np.ndarray, aug_cfg: Dict, is_train: bool) -> np.ndarray:
    if not is_train:
        return image
    if random.random() < aug_cfg["clahe_p"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if image.ndim == 2:
            image = clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if random.random() < aug_cfg["gamma_p"]:
        gamma = random.uniform(*aug_cfg["gamma_range"])
        image = np.power(image.astype(np.float32) / 255.0, gamma)
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if random.random() < aug_cfg["blur_p"]:
        kernel = max(3, int(aug_cfg["blur_kernel"]) | 1)
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    if random.random() < aug_cfg["noise_p"]:
        noise = np.random.normal(0.0, float(aug_cfg["noise_std"]) * 255.0, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def _boundary_map(binary_mask: np.ndarray) -> np.ndarray:
    binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.max() == 0:
        return np.zeros_like(binary_mask, dtype=np.float32)
    boundary = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return boundary.astype(np.float32)


def multiclass_to_boundary(mask: np.ndarray, num_classes: int) -> np.ndarray:
    boundaries = []
    for class_idx in range(num_classes):
        binary = (mask == class_idx).astype(np.uint8)
        boundaries.append(_boundary_map(binary))
    return np.stack(boundaries, axis=0)


def multilabel_to_boundary(mask: np.ndarray) -> np.ndarray:
    boundaries = []
    for class_idx in range(mask.shape[0]):
        binary = (mask[class_idx] > 0.5).astype(np.uint8)
        boundaries.append(_boundary_map(binary))
    return np.stack(boundaries, axis=0)


class XRaySegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        mode: str,
        num_classes: int,
        image_size: Tuple[int, int],
        class_names: Sequence[str],
        aug_cfg: Dict,
        is_train: bool,
        include_raw_image: bool = False,
        sample_ids: Sequence[str] | None = None,
        in_channels: int = 1,
        task_mode: str = "",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_size = tuple(image_size)
        self.class_names = list(class_names)
        self.aug_cfg = aug_cfg
        self.is_train = is_train
        self.include_raw_image = include_raw_image
        self.convert_to_multilabel = (mode == "multiclass" and task_mode in ("independent", "multilabel"))

        self.images = _list_files(self.image_dir, IMG_EXTS)
        if not self.images:
            raise RuntimeError(f"No image files found in {self.image_dir}")

        # 直接 mask 只索引当前目录下的文件，避免把 multilabel 的类别子目录文件误当成多通道 mask。
        self.direct_mask_index = _build_flat_stem_index(self.mask_dir, MASK_EXTS)
        self.class_dirs = [d for d in sorted(self.mask_dir.iterdir()) if d.is_dir()] if self.mask_dir.exists() else []
        self.class_mask_indices = {d.name: _build_stem_index(d, MASK_EXTS) for d in self.class_dirs}

        self.samples: List[dict[str, object]] = []
        for image_path in self.images:
            stem = image_path.stem
            sample: dict[str, object] = {"image": image_path}
            if self.mode == "multiclass":
                mask_path = self.direct_mask_index.get(stem)
                if mask_path is None:
                    continue
                sample["mask"] = mask_path
            elif self.mode == "multilabel":
                names = self.class_names
                channel_paths = []
                for name in names:
                    class_index = self.class_mask_indices.get(name)
                    if class_index is None or class_index.get(stem) is None:
                        channel_paths = []
                        break
                    channel_paths.append(class_index[stem])

                if channel_paths:
                    sample["mask_channels"] = channel_paths
                else:
                    direct_mask = self.direct_mask_index.get(stem)
                    if direct_mask is None:
                        continue
                    sample["mask"] = direct_mask
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            self.samples.append(sample)

        if sample_ids is not None:
            sample_id_set = set(sample_ids)
            self.samples = [sample for sample in self.samples if Path(sample["image"]).stem in sample_id_set]

        if not self.samples:
            raise RuntimeError(f"No matched samples found for {self.image_dir} and {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_multiclass_mask(self, sample: dict[str, object]) -> np.ndarray:
        mask = _read_mask(sample["mask"])  # type: ignore[index]
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = _resize_mask(mask.astype(np.int64), self.image_size).astype(np.int64)
        return np.clip(mask, 0, self.num_classes - 1)

    def _load_multilabel_mask(self, sample: dict[str, object]) -> np.ndarray:
        if "mask" in sample:
            mask = _ensure_hwc(_read_mask(sample["mask"]))  # type: ignore[index]
            mask = _resize_mask(mask, self.image_size)
            if mask.shape[-1] != self.num_classes:
                raise ValueError(f"Expected {self.num_classes} mask channels, got {mask.shape[-1]}")
            return np.transpose((mask > 0).astype(np.float32), (2, 0, 1))

        channels = []
        for path in sample["mask_channels"]:  # type: ignore[index]
            channel = _read_mask(path)
            if channel.ndim == 3:
                channel = channel[..., 0]
            channel = _resize_mask(channel, self.image_size)
            channels.append((channel > 0).astype(np.float32))

        mask = np.stack(channels, axis=0)
        if mask.shape[0] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} mask channels, got {mask.shape[0]}")
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        image_path = sample["image"]  # type: ignore[index]
        if self.in_channels == 3:
            raw_image = _resize_image(_read_rgb(image_path), self.image_size)
        else:
            raw_image = _resize_image(_read_gray(image_path), self.image_size)
        image = raw_image.copy()

        if self.mode == "multiclass":
            mask = self._load_multiclass_mask(sample)
            if self.convert_to_multilabel:
                mask = np.stack([(mask == c).astype(np.float32) for c in range(self.num_classes)], axis=0)
        else:
            mask = self._load_multilabel_mask(sample)

        if self.is_train and random.random() < self.aug_cfg["hflip_p"]:
            image = np.ascontiguousarray(np.flip(image, axis=1))
            if mask.ndim == 2:
                mask = np.ascontiguousarray(np.flip(mask, axis=1))
            else:
                mask = np.ascontiguousarray(np.flip(mask, axis=2))

        image, mask = _apply_spatial_aug(image, mask, self.aug_cfg, self.is_train)
        image = _apply_aug(image, self.aug_cfg, self.is_train)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        if image.ndim == 2:
            image = image[None, ...]
        else:
            image = np.transpose(image, (2, 0, 1))

        if self.mode == "multiclass" and not self.convert_to_multilabel:
            boundary = multiclass_to_boundary(mask, self.num_classes)
            mask_tensor = torch.from_numpy(mask).long()
        else:
            boundary = multilabel_to_boundary(mask)
            mask_tensor = torch.from_numpy(mask).float()

        result: Dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(image).float(),
            "mask": mask_tensor,
            "boundary": torch.from_numpy(boundary).float(),
            "image_id": image_path.stem,
        }
        if self.include_raw_image:
            result["raw_image"] = torch.from_numpy(raw_image).float()
        return result


def _build_common_dataset(
    cfg: Dict,
    is_train: bool,
    sample_ids: Sequence[str] | None = None,
    include_raw_image: bool = False,
) -> XRaySegmentationDataset:
    data_cfg = cfg["data"]
    return XRaySegmentationDataset(
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        mode=data_cfg["mode"],
        num_classes=data_cfg["num_classes"],
        image_size=tuple(data_cfg["image_size"]),
        class_names=data_cfg["class_names"],
        aug_cfg=cfg["augmentation"],
        is_train=is_train,
        include_raw_image=include_raw_image,
        sample_ids=sample_ids,
        in_channels=int(data_cfg.get("in_channels", 1)),
        task_mode=cfg["model"]["task_mode"],
    )


def _split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    ratio_sum = max(train_ratio + val_ratio + test_ratio, 1e-8)
    train_count = int(total * train_ratio / ratio_sum)
    val_count = int(total * val_ratio / ratio_sum)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
            test_count = max(0, test_count - 1)
        if val_count == 0:
            val_count = 1
            test_count = max(0, test_count - 1)
        if test_count == 0:
            test_count = 1
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1

    return train_count, val_count, total - train_count - val_count


def _compute_split_ids(sample_ids: Sequence[str], data_cfg: Dict) -> Dict[str, List[str]]:
    ids = sorted(sample_ids)
    rng = random.Random(int(data_cfg.get("split_seed", 42)))
    rng.shuffle(ids)

    train_count, val_count, _ = _split_counts(
        len(ids),
        float(data_cfg.get("train_ratio", 0.7)),
        float(data_cfg.get("val_ratio", 0.15)),
        float(data_cfg.get("test_ratio", 0.15)),
    )
    train_end = train_count
    val_end = train_count + val_count
    return {
        "train": ids[:train_end],
        "val": ids[train_end:val_end],
        "test": ids[val_end:],
    }


def _load_or_create_split_ids(sample_ids: Sequence[str], data_cfg: Dict) -> Dict[str, List[str]]:
    split_json = data_cfg.get("split_json", "")
    expected_ids = sorted(sample_ids)
    if split_json:
        split_path = Path(split_json)
        if split_path.exists():
            saved = json.loads(split_path.read_text(encoding="utf-8"))
            merged = sorted(saved.get("train", []) + saved.get("val", []) + saved.get("test", []))
            if merged == expected_ids:
                return {
                    "train": list(saved.get("train", [])),
                    "val": list(saved.get("val", [])),
                    "test": list(saved.get("test", [])),
                }

    split_ids = _compute_split_ids(sample_ids, data_cfg)
    if split_json:
        split_path = Path(split_json)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split_ids, indent=2), encoding="utf-8")
    return split_ids


def _compute_kfold_split_ids(
    sample_ids: Sequence[str], n_folds: int, fold_idx: int, seed: int = 42
) -> Dict[str, List[str]]:
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
    val_ids = folds[fold_idx]
    train_ids = [sid for i, fold in enumerate(folds) if i != fold_idx for sid in fold]
    return {"train": train_ids, "val": val_ids}


def build_dataset_from_config(
    cfg: Dict,
    split: str,
    is_train: bool,
    include_raw_image: bool = False,
) -> XRaySegmentationDataset:
    data_cfg = cfg["data"]

    split_json = data_cfg.get("split_json", "")
    if split_json and Path(split_json).exists():
        saved = json.loads(Path(split_json).read_text(encoding="utf-8"))
        if split in saved and saved[split]:
            if "image_dir" in data_cfg and "mask_dir" in data_cfg:
                return _build_common_dataset(
                    cfg,
                    is_train=is_train,
                    sample_ids=saved[split],
                    include_raw_image=include_raw_image,
                )
            image_dir = data_cfg.get("train_image_dir", data_cfg.get("val_image_dir", ""))
            mask_dir = data_cfg.get("train_mask_dir", data_cfg.get("val_mask_dir", ""))
            if image_dir and mask_dir:
                return XRaySegmentationDataset(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    mode=data_cfg["mode"],
                    num_classes=data_cfg["num_classes"],
                    image_size=tuple(data_cfg["image_size"]),
                    class_names=data_cfg["class_names"],
                    aug_cfg=cfg["augmentation"],
                    is_train=is_train,
                    include_raw_image=include_raw_image,
                    sample_ids=saved[split],
                    in_channels=int(data_cfg.get("in_channels", 1)),
                    task_mode=cfg["model"]["task_mode"],
                )

    image_key = f"{split}_image_dir"
    mask_key = f"{split}_mask_dir"

    if image_key in data_cfg and mask_key in data_cfg:
        return XRaySegmentationDataset(
            image_dir=data_cfg[image_key],
            mask_dir=data_cfg[mask_key],
            mode=data_cfg["mode"],
            num_classes=data_cfg["num_classes"],
            image_size=tuple(data_cfg["image_size"]),
            class_names=data_cfg["class_names"],
            aug_cfg=cfg["augmentation"],
            is_train=is_train,
            include_raw_image=include_raw_image,
            in_channels=int(data_cfg.get("in_channels", 1)),
            task_mode=cfg["model"]["task_mode"],
        )

    base_dataset = _build_common_dataset(cfg, is_train=False)
    split_ids = _load_or_create_split_ids([sample["image"].stem for sample in base_dataset.samples], data_cfg)
    return _build_common_dataset(
        cfg,
        is_train=is_train,
        sample_ids=split_ids[split],
        include_raw_image=include_raw_image,
    )


class MultiSourceDataset(Dataset):
    """Wraps multiple datasets, each with its own classes. Returns source_id per sample."""

    def __init__(self, datasets: List[Dataset], source_names: List[str]) -> None:
        self.datasets = datasets
        self.source_names = source_names
        self.cumulative = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def _locate(self, idx: int) -> Tuple[int, int]:
        for source_idx, cum in enumerate(self.cumulative):
            if idx < cum:
                local_idx = idx - (self.cumulative[source_idx - 1] if source_idx > 0 else 0)
                return source_idx, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> Dict:
        source_idx, local_idx = self._locate(idx)
        sample = self.datasets[source_idx][local_idx]
        sample["source_id"] = torch.tensor(source_idx, dtype=torch.long)
        sample["source_name"] = self.source_names[source_idx]
        return sample


class BalancedMultiSourceSampler(torch.utils.data.Sampler):
    """Samples from multiple sources, producing source-homogeneous batches.

    Each batch contains samples from only one source to avoid collation issues
    with different mask shapes. Sources are interleaved at the batch level.
    """

    def __init__(self, dataset: MultiSourceDataset, weights: List[float] | None = None, seed: int = 42, batch_size: int = 4) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.source_ranges: List[Tuple[int, int]] = []
        start = 0
        for ds in dataset.datasets:
            self.source_ranges.append((start, start + len(ds)))
            start += len(ds)
        n_sources = len(dataset.datasets)
        if weights is None:
            self.weights = [1.0 / n_sources] * n_sources
        else:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        total = len(self.dataset)
        bs = self.batch_size

        # Build per-source shuffled index lists
        source_indices = []
        for start, end in self.source_ranges:
            idxs = list(range(start, end))
            rng.shuffle(idxs)
            source_indices.append(idxs)

        # Compute how many batches each source contributes
        n_batches = total // bs
        source_batches = [max(1, int(n_batches * w)) for w in self.weights]

        # Generate batch-level interleaved indices
        all_batches = []
        for src_idx, n_b in enumerate(source_batches):
            idxs = source_indices[src_idx]
            # Repeat if needed
            while len(idxs) < n_b * bs:
                extra = list(range(self.source_ranges[src_idx][0], self.source_ranges[src_idx][1]))
                rng.shuffle(extra)
                idxs.extend(extra)
            for b in range(n_b):
                all_batches.append(idxs[b * bs:(b + 1) * bs])

        rng.shuffle(all_batches)
        indices = [idx for batch in all_batches for idx in batch]
        return iter(indices[:total])


class SingleSourceSampler(torch.utils.data.Sampler):
    """Samples only from a single source within a MultiSourceDataset."""

    def __init__(self, dataset: MultiSourceDataset, source_idx: int, seed: int = 42, batch_size: int = 4) -> None:
        self.dataset = dataset
        self.source_idx = source_idx
        self.batch_size = batch_size
        start = 0
        for i, ds in enumerate(dataset.datasets):
            if i == source_idx:
                self.start = start
                self.end = start + len(ds)
                break
            start += len(ds)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idxs = list(range(self.start, self.end))
        rng.shuffle(idxs)
        return iter(idxs)


def _build_source_dataset(
    source_cfg: Dict,
    aug_cfg: Dict,
    split: str,
    is_train: bool,
    image_size: Tuple[int, int],
    in_channels: int,
    include_raw_image: bool = False,
) -> XRaySegmentationDataset:
    image_key = f"{split}_image_dir"
    mask_key = f"{split}_mask_dir"
    task_mode = source_cfg.get("task_mode", source_cfg["mode"])
    return XRaySegmentationDataset(
        image_dir=source_cfg[image_key],
        mask_dir=source_cfg[mask_key],
        mode=source_cfg["mode"],
        num_classes=source_cfg["num_classes"],
        image_size=image_size,
        class_names=source_cfg["class_names"],
        aug_cfg=aug_cfg,
        is_train=is_train,
        include_raw_image=include_raw_image,
        in_channels=in_channels,
        task_mode=task_mode,
    )


def build_multi_source_datasets(
    cfg: Dict,
    split: str = "train",
    is_train: bool = True,
    include_raw_image: bool = False,
) -> MultiSourceDataset:
    data_cfg = cfg["data"]
    sources = data_cfg["sources"]
    image_size = tuple(data_cfg["image_size"])
    in_channels = int(data_cfg.get("in_channels", 1))
    aug_cfg = cfg["augmentation"]
    datasets = []
    names = []
    for src in sources:
        ds = _build_source_dataset(src, aug_cfg, split, is_train, image_size, in_channels, include_raw_image)
        datasets.append(ds)
        names.append(src["name"])
    return MultiSourceDataset(datasets, names)


def build_dataset_from_source_cfg(
    source_cfg: Dict,
    cfg: Dict,
    split: str = "test",
    is_train: bool = False,
    include_raw_image: bool = False,
) -> XRaySegmentationDataset:
    """Build a single-source dataset from a source config entry."""
    data_cfg = cfg["data"]
    image_size = tuple(data_cfg["image_size"])
    in_channels = int(data_cfg.get("in_channels", 1))
    aug_cfg = cfg["augmentation"]
    return _build_source_dataset(source_cfg, aug_cfg, split, is_train, image_size, in_channels, include_raw_image)


def build_train_val_datasets(cfg: Dict) -> tuple[XRaySegmentationDataset, XRaySegmentationDataset]:
    return (
        build_dataset_from_config(cfg, split="train", is_train=True),
        build_dataset_from_config(cfg, split="val", is_train=False),
    )
