"""Test-Time Augmentation (TTA) for segmentation inference."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def tta_predict(
    model: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    image: torch.Tensor,
    scales: tuple[float, ...] = (1.0,),
    flip_horizontal: bool = True,
    flip_vertical: bool = False,
) -> torch.Tensor:
    """Run TTA and return averaged logits.

    Args:
        model: callable that takes [B,C,H,W] and returns dict with 'seg_logits'
        image: input tensor [B, C, H, W]
        scales: tuple of scale factors (1.0 = original)
        flip_horizontal: include horizontal flip
        flip_vertical: include vertical flip

    Returns:
        Averaged logits [B, num_classes, H, W]
    """
    B, C, H, W = image.shape
    logits_sum = None
    count = 0

    flip_modes: list[tuple[bool, bool]] = [(False, False)]
    if flip_horizontal:
        flip_modes.append((True, False))
    if flip_vertical:
        flip_modes.append((False, True))
    if flip_horizontal and flip_vertical:
        flip_modes.append((True, True))

    for scale in scales:
        if scale != 1.0:
            sH, sW = int(H * scale), int(W * scale)
            scaled = F.interpolate(image, size=(sH, sW), mode='bilinear', align_corners=False)
        else:
            scaled = image

        for do_hflip, do_vflip in flip_modes:
            x = scaled
            if do_hflip:
                x = torch.flip(x, dims=[-1])
            if do_vflip:
                x = torch.flip(x, dims=[-2])

            out = model(x)['seg_logits']

            if do_vflip:
                out = torch.flip(out, dims=[-2])
            if do_hflip:
                out = torch.flip(out, dims=[-1])

            if out.shape[-2:] != (H, W):
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

            if logits_sum is None:
                logits_sum = torch.zeros_like(out)
            logits_sum += out
            count += 1

    return logits_sum / count
