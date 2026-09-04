"""Pretrained encoder wrappers using timm for ImageNet-initialized backbones.

These encoders output 4-level feature maps matching the PyramidTransformerEncoder interface:
    F1: [B, C1, H/4,  W/4]
    F2: [B, C2, H/8,  W/8]
    F3: [B, C3, H/16, W/16]
    F4: [B, C4, H/32, W/32]
"""
from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import List
import warnings

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    safe_load_file = None


ENCODER_REGISTRY = {
    "pvt_v2_b0": "pvt_v2_b0",
    "pvt_v2_b1": "pvt_v2_b1",
    "pvt_v2_b2": "pvt_v2_b2",
    "pvt_v2_b3": "pvt_v2_b3",
    "pvt_v2_b4": "pvt_v2_b4",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "efficientnet_b4": "efficientnet_b4",
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "maxvit_tiny": "maxvit_tiny_tf_224",
}


class PretrainedEncoder(nn.Module):
    """Wraps a timm feature extractor to output 4-level pyramid features.

    Handles:
    - Grayscale input adaptation (1-ch → 3-ch via learned projection)
    - Channel count exposure via .out_channels for downstream projection
    """

    def __init__(
        self,
        encoder_name: str = "pvt_v2_b2",
        in_channels: int = 1,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for pretrained encoders: pip install timm")

        timm_name = ENCODER_REGISTRY.get(encoder_name, encoder_name)
        # Models with stem level (ResNet, EfficientNet) need (1,2,3,4) for stride 4/8/16/32
        # Models with exactly 4 levels (PVT, Swin, ConvNeXt, MaxViT) use (0,1,2,3)
        _skip_stem = encoder_name in ("resnet50", "efficientnet_b4")
        out_indices = (1, 2, 3, 4) if _skip_stem else (0, 1, 2, 3)

        if checkpoint_path:
            self.backbone = timm.create_model(
                timm_name,
                pretrained=False,
                features_only=True,
                out_indices=out_indices,
            )
            state_dict = _load_checkpoint_state_dict(checkpoint_path)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            if missing_keys or unexpected_keys:
                warnings.warn(
                    f"Loaded local checkpoint for '{timm_name}' with {len(missing_keys)} missing and "
                    f"{len(unexpected_keys)} unexpected keys.",
                    RuntimeWarning,
                )
        else:
            try:
                self.backbone = timm.create_model(
                    timm_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=out_indices,
                )
            except Exception as exc:
                if not pretrained:
                    raise
                warnings.warn(
                    f"Failed to load pretrained weights for '{timm_name}' ({exc}). "
                    "Falling back to random initialization so training can continue.",
                    RuntimeWarning,
                )
                self.backbone = timm.create_model(
                    timm_name,
                    pretrained=False,
                    features_only=True,
                    out_indices=out_indices,
                )
        feat_info = self.backbone.feature_info.channels()
        self.out_channels = feat_info

        self.input_adapter = None
        if in_channels != 3:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
                nn.GroupNorm(1, 3),
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        feats = self.backbone(x)
        return feats


def build_encoder(cfg: dict) -> tuple[nn.Module, list[int]]:
    """Build encoder based on config. Returns (encoder, out_channels_list).

    Config options:
        model.encoder_type: "pyramid_transformer" (default) or any key in ENCODER_REGISTRY
        model.encoder_pretrained: true/false (default: true)
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    encoder_type = model_cfg.get("encoder_type", "pyramid_transformer")
    in_channels = int(data_cfg.get("in_channels", 1))

    if encoder_type == "pyramid_transformer":
        from .pyramid_transformer_encoder import PyramidTransformerEncoder
        encoder = PyramidTransformerEncoder(
            in_channels=in_channels,
            embed_dim=int(model_cfg["embed_dim"]),
            depths=tuple(model_cfg["depths"]),
            num_heads=tuple(model_cfg["num_heads_pyramid"]),
            window_size=int(model_cfg["window_size"]),
        )
        return encoder, encoder.out_channels

    pretrained = bool(model_cfg.get("encoder_pretrained", True))
    checkpoint_path = model_cfg.get("encoder_checkpoint_path")
    encoder = PretrainedEncoder(
        encoder_name=encoder_type,
        in_channels=in_channels,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )
    return encoder, encoder.out_channels


def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        if safe_load_file is not None:
            state_dict = safe_load_file(str(path))
        else:
            state_dict = _load_safetensors_fallback(path)
    else:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if isinstance(checkpoint.get("state_dict"), dict):
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint.get("model"), dict):
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint.get("model_state_dict"), dict):
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise TypeError(f"Unsupported checkpoint format at: {path}")

    return _strip_wrappers(state_dict)


def _load_safetensors_fallback(path: Path) -> dict:
    dtype_map = {
        "BOOL": np.bool_,
        "U8": np.uint8,
        "I8": np.int8,
        "I16": np.int16,
        "U16": np.uint16,
        "I32": np.int32,
        "U32": np.uint32,
        "I64": np.int64,
        "U64": np.uint64,
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
    }

    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        payload = f.read()

    state_dict = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue

        dtype_name = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        tensor_bytes = memoryview(payload)[start:end]

        if dtype_name == "BF16":
            raw = np.frombuffer(tensor_bytes, dtype=np.uint16).copy()
            tensor = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
        else:
            np_dtype = dtype_map.get(dtype_name)
            if np_dtype is None:
                raise ValueError(f"Unsupported safetensors dtype '{dtype_name}' in {path}")
            array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape).copy()
            tensor = torch.from_numpy(array)

        if shape:
            tensor = tensor.reshape(shape)
        state_dict[key] = tensor

    warnings.warn(
        "Loaded .safetensors checkpoint without the optional `safetensors` package. "
        "Install `safetensors` for faster and more robust loading.",
        RuntimeWarning,
    )
    return state_dict


def _strip_wrappers(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]
        cleaned[new_key] = value
    return cleaned
