from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def _binary_target_from_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return (mask > 0).float().unsqueeze(1)
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            return mask.float()
        return mask[:, 1:].amax(dim=1, keepdim=True).float()
    raise ValueError(f'Unsupported binary target shape: {tuple(mask.shape)}')


def _single_boundary_target(boundary: torch.Tensor) -> torch.Tensor:
    if boundary.ndim == 3:
        return boundary.float().unsqueeze(1)
    if boundary.ndim == 4:
        if boundary.shape[1] == 1:
            return boundary.float()
        return boundary[:, 1:].amax(dim=1, keepdim=True).float()
    raise ValueError(f'Unsupported boundary shape: {tuple(boundary.shape)}')


def _aggregate_boundary_target(boundary: torch.Tensor, has_background_channel: bool) -> torch.Tensor:
    if boundary.ndim == 3:
        return boundary.float().unsqueeze(1)
    if boundary.ndim == 4:
        if boundary.shape[1] == 1:
            return boundary.float()
        start_idx = 1 if has_background_channel and boundary.shape[1] > 1 else 0
        return boundary[:, start_idx:].amax(dim=1, keepdim=True).float()
    raise ValueError(f'Unsupported boundary shape: {tuple(boundary.shape)}')


def _edge_target(boundary: torch.Tensor, channels: int) -> torch.Tensor:
    if channels == 1:
        return _single_boundary_target(boundary)
    if boundary.ndim == 3:
        return F.one_hot(boundary.long(), num_classes=channels).permute(0, 3, 1, 2).float()
    if boundary.ndim == 4:
        if boundary.shape[1] == channels:
            return boundary.float()
        if boundary.shape[1] > channels:
            return boundary[:, :channels].float()
    raise ValueError(f'Unsupported edge target shape {tuple(boundary.shape)} for {channels} channels')


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    inter = (probs * target).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def binary_prob_dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    target = target.float()
    inter = (probs * target).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool,
    eps: float,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    tgt = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * tgt).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + tgt.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    if ignore_background and num_classes > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()


def multilabel_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    inter = (probs * target).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def boundary_bce_dice_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float, from_logits: bool = True) -> torch.Tensor:
    if target.shape[-2:] != prediction.shape[-2:]:
        target = F.interpolate(target, size=prediction.shape[-2:], mode='nearest')
    target = target.float()
    if from_logits:
        bce = F.binary_cross_entropy_with_logits(prediction, target)
        dice = binary_dice_loss(prediction, target, eps)
    else:
        probs = prediction.float().clamp(0.0, 1.0)
        with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
            bce = F.binary_cross_entropy(probs, target.float())
            dice = binary_prob_dice_loss(probs, target.float(), eps)
    return bce + dice


def compute_distance_map(mask: torch.Tensor) -> torch.Tensor:
    """Compute signed distance map from binary mask. Positive inside, negative outside."""
    mask_np = mask.detach().cpu().numpy()
    B, C = mask_np.shape[:2]
    dist_maps = np.zeros_like(mask_np, dtype=np.float32)
    for b in range(B):
        for c in range(C):
            m = mask_np[b, c]
            if m.sum() == 0:
                dist_maps[b, c] = -ndimage.distance_transform_edt(1 - m)
            elif m.sum() == m.size:
                dist_maps[b, c] = ndimage.distance_transform_edt(m)
            else:
                pos_dist = ndimage.distance_transform_edt(m)
                neg_dist = ndimage.distance_transform_edt(1 - m)
                dist_maps[b, c] = pos_dist - neg_dist
    return torch.from_numpy(dist_maps).to(mask.device, dtype=mask.dtype)


def boundary_distance_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Boundary loss: penalizes FP far from object and FN deep inside object.

    Uses signed distance where positive = inside object, negative = outside.
    Distance is normalized to [0, 1] per sample so magnitude is comparable to dice loss.
    """
    dist_map = compute_distance_map(target)
    abs_dist = dist_map.abs()
    max_dist = abs_dist.flatten(2).max(dim=-1).values.view(-1, abs_dist.shape[1], 1, 1).clamp(min=1.0)
    abs_dist = abs_dist / max_dist
    probs = torch.sigmoid(logits)
    target_f = target.float()
    fp_penalty = (1.0 - target_f) * probs * abs_dist
    fn_penalty = target_f * (1.0 - probs) * abs_dist
    return (fp_penalty + fn_penalty).mean()


def boundary_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    boundary_width: int = 5,
    curvature_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss concentrated on boundary regions with optional curvature weighting.

    Near the boundary, prediction errors matter most for HD95.
    This loss applies focal weighting (hard example mining) only within
    a narrow band around the GT boundary, forcing the model to focus
    on the hardest boundary pixels.

    Args:
        logits: [B, C, H, W] raw predictions
        target: [B, C, H, W] binary masks
        gamma: focal exponent — higher = more focus on hard pixels
        boundary_width: dilation radius defining the boundary band (pixels)
        curvature_weight: [B, C, H, W] optional per-pixel curvature magnitude for extra weighting
    """
    C = logits.shape[1]
    target_f = target.float()
    probs = torch.sigmoid(logits)

    # Extract boundary band via morphological dilation - erosion
    k = 2 * boundary_width + 1
    morph_kernel = torch.ones(1, 1, k, k, device=logits.device)
    pad = boundary_width
    dilated = F.conv2d(
        F.pad(target_f, [pad] * 4, mode='replicate'),
        morph_kernel.repeat(C, 1, 1, 1), groups=C,
    ).clamp(0, 1)
    eroded = 1.0 - F.conv2d(
        F.pad(1.0 - target_f, [pad] * 4, mode='replicate'),
        morph_kernel.repeat(C, 1, 1, 1), groups=C,
    ).clamp(0, 1)
    boundary_band = (dilated - eroded).clamp(0, 1)

    # Focal BCE within boundary band
    bce = F.binary_cross_entropy_with_logits(logits, target_f, reduction='none')
    pt = probs * target_f + (1 - probs) * (1 - target_f)
    focal_weight = (1 - pt).pow(gamma)
    focal_bce = focal_weight * bce * boundary_band

    # Curvature weighting: high-curvature regions (corners, tips) get extra focus
    if curvature_weight is not None:
        curv_w = curvature_weight.detach()
        if curv_w.shape[-2:] != logits.shape[-2:]:
            curv_w = F.interpolate(curv_w, size=logits.shape[-2:], mode='bilinear', align_corners=False)
        curv_boost = 1.0 + curv_w.clamp(0, 3.0)
        focal_bce = focal_bce * curv_boost

    # Normalize by boundary area to keep loss scale stable
    boundary_area = boundary_band.sum().clamp(min=1.0)
    return focal_bce.sum() / boundary_area


def direction_field_loss(
    pred_direction: torch.Tensor,
    gt_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Supervise direction field with GT SDM gradient direction.

    Computes GT boundary normal direction from mask boundaries,
    then penalizes angular deviation of predicted direction field.
    """
    if pred_direction is None or num_classes <= 1:
        return gt_mask.new_zeros(())

    target_f = gt_mask.float()
    if target_f.ndim == 3:
        target_f = target_f.unsqueeze(1)

    # Compute GT boundary normals from mask using Sobel
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=gt_mask.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=gt_mask.device).reshape(1, 1, 3, 3)

    # Use first class channel as reference for direction
    ref = target_f[:, 0:1]
    gx = F.conv2d(ref, sobel_x, padding=1)
    gy = F.conv2d(ref, sobel_y, padding=1)
    gt_dir = torch.cat([gx, gy], dim=1)
    gt_dir_norm = F.normalize(gt_dir, p=2, dim=1, eps=1e-6)

    # Only supervise at boundary pixels (where gradient magnitude > threshold)
    grad_mag = (gx.square() + gy.square()).sqrt()
    boundary_mask = (grad_mag > 0.1).float()

    if boundary_mask.sum() < 1.0:
        return gt_mask.new_zeros(())

    # Resize pred to match
    if pred_direction.shape[-2:] != gt_dir_norm.shape[-2:]:
        pred_direction = F.interpolate(pred_direction, size=gt_dir_norm.shape[-2:], mode='bilinear', align_corners=False)
        pred_direction = F.normalize(pred_direction, p=2, dim=1, eps=1e-6)

    # Cosine similarity loss at boundary pixels
    cos_sim = (pred_direction * gt_dir_norm).sum(dim=1, keepdim=True)
    loss = (1.0 - cos_sim) * boundary_mask
    return loss.sum() / boundary_mask.sum().clamp(min=1.0)


def topology_critical_loss(
    seg_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    relation_map: torch.Tensor,
    sdm_pred: torch.Tensor,
    relation_types: list,
    tau: float = 2.0,
) -> torch.Tensor:
    """Topology-aware critical loss: penalize segmentation errors where predicted
    inter-class topology violates ground truth, weighted by boundary proximity.

    Args:
        seg_logits: [B, C, H, W] raw logits
        gt_mask: [B, C, H, W] binary ground truth
        relation_map: [B, 3, h, w] predicted topology from ICDC (disjoint/intersection/containment)
        sdm_pred: [B, C, H, W] predicted signed distance map
        relation_types: list of active relation types
        tau: temperature for boundary distance decay
    """
    B, C, H, W = seg_logits.shape
    if C < 2:
        return seg_logits.new_zeros(())

    gt_f = gt_mask.float()
    if gt_f.shape[-2:] != (H, W):
        gt_f = F.interpolate(gt_f, size=(H, W), mode='nearest')

    # Compute GT relation at relation_map resolution
    h, w = relation_map.shape[-2:]
    gt_small = F.interpolate(gt_f, size=(h, w), mode='nearest')

    # GT disjoint: product of any two class masks should be 0
    gt_overlap = torch.zeros(B, 1, h, w, device=seg_logits.device)
    for i in range(C):
        for j in range(i + 1, C):
            gt_overlap = gt_overlap + (gt_small[:, i:i+1] * gt_small[:, j:j+1])
    gt_disjoint = (gt_overlap < 0.5).float()
    gt_intersect = 1.0 - gt_disjoint

    # Build GT relation tensor [B, 3, h, w]: [disjoint, intersection, containment]
    gt_relation = torch.stack([gt_disjoint.squeeze(1), gt_intersect.squeeze(1), gt_intersect.squeeze(1)], dim=1)

    # Mismatch between predicted and GT topology
    pred_relation = torch.sigmoid(relation_map)
    mismatch = (pred_relation - gt_relation).abs().mean(dim=1, keepdim=True)

    # Boundary weight from SDM: exp(-|sdm| / tau)
    sdm_abs = sdm_pred.abs().mean(dim=1, keepdim=True)
    boundary_weight = torch.exp(-sdm_abs / tau)

    # Upsample mismatch to full resolution
    if mismatch.shape[-2:] != (H, W):
        mismatch = F.interpolate(mismatch, size=(H, W), mode='bilinear', align_corners=False)

    # Critical attention = topology mismatch × boundary proximity
    critical_attention = mismatch * boundary_weight

    # Weighted BCE on critical regions
    bce = F.binary_cross_entropy_with_logits(seg_logits, gt_f, reduction='none')
    weighted_loss = (bce * critical_attention).sum() / critical_attention.sum().clamp(min=1.0)
    return weighted_loss


def adjacency_boundary_loss(
    seg_logits: torch.Tensor,
    sdm_pred: torch.Tensor,
    adjacency_pairs: list[list[int]],
    boundary_width: float = 3.0,
) -> torch.Tensor:
    """Enforce boundary consistency between anatomically adjacent classes.

    At the contact zone of adjacent classes (e.g., heart-lung), their predictions
    should be complementary: P(class_i) + P(class_j) ≈ 1, and their SDMs should
    be mirror images (sdm_i ≈ -sdm_j).

    Args:
        seg_logits: [B, C, H, W]
        sdm_pred: [B, C, H, W]
        adjacency_pairs: list of [i, j] class index pairs that share a boundary
        boundary_width: SDM distance threshold defining the contact zone
    """
    if not adjacency_pairs or seg_logits.shape[1] < 2:
        return seg_logits.new_zeros(())

    probs = torch.sigmoid(seg_logits)
    total_loss = seg_logits.new_zeros(())
    count = 0

    for pair in adjacency_pairs:
        i, j = int(pair[0]), int(pair[1])
        if i >= seg_logits.shape[1] or j >= seg_logits.shape[1]:
            continue

        # Contact zone: where both classes are near their boundaries
        near_i = (sdm_pred[:, i:i+1].abs() < boundary_width).float()
        near_j = (sdm_pred[:, j:j+1].abs() < boundary_width).float()
        contact_zone = near_i * near_j

        if contact_zone.sum() < 1.0:
            continue

        # Complementarity: P(i) + P(j) should be close to 1 at contact zone
        complement_err = (probs[:, i:i+1] + probs[:, j:j+1] - 1.0).abs()
        complement_loss = (complement_err * contact_zone).sum() / contact_zone.sum().clamp(min=1.0)

        # SDM mirror: sdm_i + sdm_j ≈ 0 at contact zone (one positive, one negative)
        sdm_mirror_err = (sdm_pred[:, i:i+1] + sdm_pred[:, j:j+1]).abs()
        sdm_mirror_loss = (sdm_mirror_err * contact_zone).sum() / contact_zone.sum().clamp(min=1.0)

        total_loss = total_loss + complement_loss + 0.5 * sdm_mirror_loss
        count += 1

    return total_loss / max(count, 1)


def occlusion_aware_weight(
    gt_mask: torch.Tensor,
    containment_hierarchy: list[list[int]],
) -> torch.Tensor:
    """Generate per-pixel loss weight that accounts for anatomical occlusion.

    In X-ray projection imaging, structures overlap. When a "child" class (e.g., clavicle)
    overlaps a "parent" class (e.g., lung), the parent should still be considered present
    beneath the child. This function reduces loss penalty on the parent class in regions
    where the child class is present, preventing the parent from being "cut" by the child.

    Args:
        gt_mask: [B, C, H, W] ground truth (multilabel)
        containment_hierarchy: list of [parent_idx, child_idx] pairs
            e.g., [[0, 2]] means class 0 (lung) contains class 2 (clavicle)

    Returns:
        weight: [B, C, H, W] per-pixel loss weight (1.0 = normal, <1.0 = reduced penalty)
    """
    weight = torch.ones_like(gt_mask)
    for pair in containment_hierarchy:
        parent_idx, child_idx = int(pair[0]), int(pair[1])
        if parent_idx >= gt_mask.shape[1] or child_idx >= gt_mask.shape[1]:
            continue
        child_region = gt_mask[:, child_idx:child_idx+1]
        # In child's region, reduce loss weight on parent class
        # This allows the model to predict parent=1 even where GT parent=0 (occluded)
        weight[:, parent_idx:parent_idx+1] = weight[:, parent_idx:parent_idx+1] * (1.0 - 0.7 * child_region)
    return weight


def compute_normalized_sdm(mask: torch.Tensor) -> torch.Tensor:
    """Compute signed distance map normalized to [-1, 1]. Positive inside, negative outside."""
    dist_map = compute_distance_map(mask)
    max_dist = dist_map.abs().flatten(2).max(dim=-1).values.view(-1, dist_map.shape[1], 1, 1).clamp(min=1.0)
    return dist_map / max_dist


def sdm_loss(sdm_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Signed Distance Map supervision loss (smooth L1)."""
    sdm_gt = compute_normalized_sdm(target)
    return F.smooth_l1_loss(sdm_pred, sdm_gt)


def boundary_smoothness_loss(seg_logits: torch.Tensor, target: torch.Tensor, curvature_budgets: torch.Tensor) -> torch.Tensor:
    """Penalize non-smooth boundaries in segmentation predictions.

    Computes Laplacian of sigmoid(logits) at GT boundary locations.
    High curvature at boundaries = jagged edges = penalized.
    Per-class budgets control how much curvature is tolerated.
    """
    C = seg_logits.shape[1]
    probs = torch.sigmoid(seg_logits)
    laplacian_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=seg_logits.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    curvature = F.conv2d(probs, laplacian_kernel, padding=1, groups=C)
    # GT boundary mask via morphological gradient
    morph_kernel = torch.ones(1, 1, 3, 3, device=target.device)
    target_f = target.float()
    dilated = F.conv2d(F.pad(target_f, [1]*4, mode='replicate'), morph_kernel.repeat(C, 1, 1, 1), groups=C).clamp(0, 1)
    eroded = 1.0 - F.conv2d(F.pad(1.0 - target_f, [1]*4, mode='replicate'), morph_kernel.repeat(C, 1, 1, 1), groups=C).clamp(0, 1)
    gt_boundary = (dilated - eroded).clamp(0, 1)
    curvature_at_boundary = curvature.abs() * gt_boundary
    budgets = curvature_budgets.to(seg_logits.device).view(1, -1, 1, 1)
    excess = F.relu(curvature_at_boundary - budgets)
    return excess.mean()


def gate_entropy_loss(gate_activations: list[torch.Tensor]) -> torch.Tensor:
    """Encourage gates to stay away from 0/1 (prevent collapse/saturation)."""
    if not gate_activations:
        return torch.zeros(1, device='cpu')
    eps = 1e-3
    losses = []
    for g in gate_activations:
        g_clamped = g.float().clamp(eps, 1.0 - eps)
        entropy = -(g_clamped * g_clamped.log() + (1 - g_clamped) * (1 - g_clamped).log())
        losses.append(entropy.mean())
    return -torch.stack(losses).mean()


def gate_diversity_loss(gate_activations: list[torch.Tensor]) -> torch.Tensor:
    """Penalize similarity between different gate spatial patterns.

    Pools each gate to a fixed-size descriptor for efficient pairwise comparison.
    """
    if len(gate_activations) < 2:
        return torch.zeros(1, device=gate_activations[0].device if gate_activations else 'cpu')
    target_size = 16
    descriptors = []
    for g in gate_activations:
        pooled = F.adaptive_avg_pool2d(g.float(), (4, 4))
        desc = pooled.mean(dim=1).flatten(1)
        descriptors.append(desc)
    loss = torch.zeros(1, device=descriptors[0].device)
    count = 0
    for i in range(len(descriptors)):
        for j in range(i + 1, len(descriptors)):
            di = descriptors[i] / (descriptors[i].norm(dim=1, keepdim=True) + 1e-8)
            dj = descriptors[j] / (descriptors[j].norm(dim=1, keepdim=True) + 1e-8)
            loss = loss + (di * dj).sum(dim=1).abs().mean()
            count += 1
    return loss / max(count, 1)


class CompositeSegmentationLoss(nn.Module):
    _TASK_MODE_MAP = {'binary': 'exclusive', 'multiclass': 'exclusive', 'multilabel': 'independent'}

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        raw_task_mode = cfg['model']['task_mode']
        self.task_mode = self._TASK_MODE_MAP.get(raw_task_mode, raw_task_mode)
        self.data_mode = cfg['data']['mode']
        self.num_classes = int(cfg['model']['num_classes'])
        self.seg_classes = 1 if raw_task_mode == 'binary' else self.num_classes
        self.ignore_background = cfg['data'].get('ignore_background', True)
        self.loss_cfg = cfg['loss']
        ablation_cfg = cfg['model'].get('ablation', {})
        self.use_plain_unet_baseline = bool(ablation_cfg.get('use_plain_unet_baseline', False))
        self.disable_edge_branch = bool(ablation_cfg.get('disable_edge_branch', False)) or self.use_plain_unet_baseline
        self.disable_overlap_prior = bool(ablation_cfg.get('disable_overlap_prior', False)) or self.use_plain_unet_baseline
        self.disable_boundary_generators = bool(ablation_cfg.get('disable_boundary_generators', False)) or self.use_plain_unet_baseline
        self.disable_region_to_edge_constraint = bool(ablation_cfg.get('disable_region_to_edge_constraint', False)) or self.use_plain_unet_baseline
        self.eps = float(self.loss_cfg['eps'])
        pos_weight_cfg = self.loss_cfg.get('pos_weight', 1.0)
        if pos_weight_cfg == 'auto':
            self.register_buffer('pos_weight', None)
            self.register_buffer('class_weight', None)
        elif isinstance(pos_weight_cfg, (list, tuple)):
            self.register_buffer('pos_weight', torch.tensor(pos_weight_cfg).float().view(1, -1, 1, 1))
            self.register_buffer('class_weight', None)
        else:
            self.register_buffer('pos_weight', torch.tensor([float(pos_weight_cfg)]).view(1, 1, 1, 1))
            self.register_buffer('class_weight', None)

    def compute_pos_weight_from_dataset(self, dataset, max_samples: int = 200) -> None:
        """Auto-compute class balancing weights from training data."""
        import numpy as np
        device = next((p.device for p in self.parameters()), torch.device('cpu')) if len(list(self.parameters())) > 0 else (self.pos_weight.device if self.pos_weight is not None else torch.device('cpu'))
        n = min(len(dataset), max_samples)
        if self.task_mode == 'exclusive' and self.seg_classes > 1:
            class_counts = np.zeros(self.seg_classes, dtype=np.float64)
            for i in range(n):
                mask = dataset[i]['mask'].numpy()
                if mask.ndim == 3:
                    mask = mask.argmax(axis=0)
                for c in range(self.seg_classes):
                    class_counts[c] += (mask == c).sum()
            total = class_counts.sum()
            weights = np.clip(total / (self.seg_classes * class_counts + 1e-6), 0.5, 20.0)
            if self.ignore_background and self.seg_classes > 1:
                weights[0] = weights[1:].min() * 0.3
            self.register_buffer('class_weight', torch.tensor(weights).float().to(device))
            print(f"Auto class_weight: {[f'{w:.2f}' for w in weights.tolist()]}")
        else:
            num_classes = self.seg_classes
            pos_counts = np.zeros(num_classes, dtype=np.float64)
            total_pixels = 0.0
            for i in range(n):
                mask = dataset[i]['mask'].numpy()
                if num_classes == 1:
                    pos_counts[0] += mask.sum()
                    total_pixels += mask.size
                else:
                    for c in range(min(num_classes, mask.shape[0])):
                        pos_counts[c] += mask[c].sum()
                    total_pixels += mask[0].size
            neg_counts = total_pixels - pos_counts
            weights = np.clip(neg_counts / (pos_counts + 1e-6), 1.0, 100.0)
            weights = np.sqrt(weights)
            self.register_buffer('pos_weight', torch.tensor(weights).float().view(1, -1, 1, 1).to(device))
            print(f"Auto pos_weight: {[f'{w:.2f}' for w in weights.tolist()]}")

    def _segmentation_loss(self, seg_logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pw = self.pos_weight.to(seg_logits.device) if self.pos_weight is not None else None
        cw = self.class_weight.to(seg_logits.device) if self.class_weight is not None else None
        if self.task_mode == 'exclusive':
            if self.seg_classes == 1:
                target = _binary_target_from_mask(mask)
                seg_primary = F.binary_cross_entropy_with_logits(seg_logits, target, pos_weight=pw)
                seg_dice = binary_dice_loss(seg_logits, target, self.eps)
                return seg_primary, seg_dice
            seg_primary = F.cross_entropy(seg_logits, mask.long(), weight=cw)
            seg_dice = multiclass_dice_loss(seg_logits, mask, self.num_classes, self.ignore_background, self.eps)
            return seg_primary, seg_dice
        # independent (multilabel)
        target = mask.float()
        bce = F.binary_cross_entropy_with_logits(seg_logits, target, pos_weight=pw, reduction='none')
        # Occlusion-aware weighting: reduce penalty on parent class in child's region
        containment_hierarchy = self.loss_cfg.get('containment_hierarchy', [])
        if containment_hierarchy:
            occ_weight = occlusion_aware_weight(target, containment_hierarchy)
            bce = bce * occ_weight
        # Anatomy-aware weighting: upweight boundary regions per class
        boundary_boost = self.loss_cfg.get('anatomy_boundary_boost', None)
        if boundary_boost is not None and isinstance(boundary_boost, list):
            boost_tensor = torch.tensor(boundary_boost, device=seg_logits.device, dtype=seg_logits.dtype)
            boost_tensor = boost_tensor.view(1, -1, 1, 1)
            boundary_mask = self._extract_boundary_band(target, width=3)
            weight_map = 1.0 + boundary_mask * (boost_tensor - 1.0)
            bce = bce * weight_map
        seg_primary = bce.mean()
        seg_dice = multilabel_dice_loss(seg_logits, target, self.eps)
        return seg_primary, seg_dice

    @staticmethod
    def _extract_boundary_band(target: torch.Tensor, width: int = 3) -> torch.Tensor:
        kernel = torch.ones(1, 1, width, width, device=target.device, dtype=target.dtype)
        bands = []
        for c in range(target.shape[1]):
            ch = target[:, c:c+1]
            dilated = F.conv2d(ch, kernel, padding=width // 2).clamp(0, 1)
            eroded = 1.0 - F.conv2d(1.0 - ch, kernel, padding=width // 2).clamp(0, 1)
            bands.append(dilated - eroded)
        return torch.cat(bands, dim=1).clamp(0, 1)

    @staticmethod
    def _cross_scale_edge_consistency(edge_logits: list[torch.Tensor]) -> torch.Tensor:
        loss = edge_logits[0].new_zeros(())
        pairs = 0
        for i in range(len(edge_logits) - 1):
            hi = edge_logits[i]
            lo = edge_logits[i + 1]
            hi_down = F.interpolate(hi, size=lo.shape[-2:], mode='bilinear', align_corners=False)
            loss = loss + F.l1_loss(hi_down, lo)
            pairs += 1
        return loss / max(pairs, 1)

    def _uncertainty_calibration_loss(self, seg_logits: torch.Tensor, mask: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """Uncertainty should be high where prediction error is high."""
        with torch.no_grad():
            if self.task_mode == 'exclusive' and self.seg_classes > 1:
                pred = seg_logits.argmax(dim=1)
                target = mask.long() if mask.ndim == 3 else mask.argmax(dim=1)
                error = (pred != target).float().unsqueeze(1)
            else:
                target = mask.float()
                if target.ndim == 3:
                    target = target.unsqueeze(1)
                pred = (torch.sigmoid(seg_logits) > 0.5).float()
                error = (pred != target).float().amax(dim=1, keepdim=True)
        u = uncertainty.float().clamp(1e-6, 1.0 - 1e-6)
        error = error.to(u.dtype)
        return -(error * u.log() + (1.0 - error) * (1.0 - u).log()).mean()

    def _consistency_loss(
        self,
        edge_attention_intersection: torch.Tensor,
        edge_attention_inner: torch.Tensor,
        edge_attention_disjoint: torch.Tensor | None,
        shared_prior: torch.Tensor,
        inner_prior: torch.Tensor,
        disjoint_prior: torch.Tensor | None,
    ) -> torch.Tensor:
        shared_up = F.interpolate(shared_prior, size=edge_attention_intersection.shape[-2:], mode='bilinear', align_corners=False)
        inner_up = F.interpolate(inner_prior, size=edge_attention_inner.shape[-2:], mode='bilinear', align_corners=False)
        if edge_attention_intersection.shape[1] > 1:
            shared_up = shared_up.expand(-1, edge_attention_intersection.shape[1], -1, -1)
        if edge_attention_inner.shape[1] > 1:
            inner_up = inner_up.expand(-1, edge_attention_inner.shape[1], -1, -1)

        loss = F.l1_loss(edge_attention_intersection, shared_up) + F.l1_loss(edge_attention_inner, inner_up)

        if edge_attention_disjoint is not None and disjoint_prior is not None:
            disjoint_up = F.interpolate(disjoint_prior, size=edge_attention_disjoint.shape[-2:], mode='bilinear', align_corners=False)
            if edge_attention_disjoint.shape[1] > 1:
                disjoint_up = disjoint_up.expand(-1, edge_attention_disjoint.shape[1], -1, -1)
            loss = loss + F.l1_loss(edge_attention_disjoint, disjoint_up)

        return loss

    def forward(self, preds: Dict[str, torch.Tensor | list[torch.Tensor]], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seg_logits = preds['seg_logits']
        seg_primary, seg_dice = self._segmentation_loss(seg_logits, batch['mask'])
        lambda_ce = float(self.loss_cfg.get('lambda_ce', 1.0))
        lambda_dice = float(self.loss_cfg.get('lambda_dice', 1.0))
        seg_loss = lambda_ce * seg_primary + lambda_dice * seg_dice

        # Auxiliary deep supervision loss (d3)
        aux_logits = preds.get('aux_seg_logits')
        if aux_logits is not None:
            aux_primary, aux_dice = self._segmentation_loss(aux_logits, batch['mask'])
            aux_loss = lambda_ce * aux_primary + lambda_dice * aux_dice
        else:
            aux_loss = seg_logits.new_zeros(())

        # Auxiliary deep supervision loss (d2)
        aux_logits_d2 = preds.get('aux_seg_logits_d2')
        if aux_logits_d2 is not None:
            aux2_primary, aux2_dice = self._segmentation_loss(aux_logits_d2, batch['mask'])
            aux_loss_d2 = lambda_ce * aux2_primary + lambda_dice * aux2_dice
        else:
            aux_loss_d2 = seg_logits.new_zeros(())

        boundary = batch.get('boundary', batch['mask'])
        if self.disable_edge_branch:
            edge_loss = seg_logits.new_zeros(())
        else:
            edge_logits = preds['edge_logits']
            edge_losses = []
            for edge_logit in edge_logits:
                edge_target = _edge_target(boundary, edge_logit.shape[1])
                edge_losses.append(boundary_bce_dice_loss(edge_logit, edge_target, self.eps, from_logits=True))
            edge_loss = torch.stack(edge_losses).mean()

        has_background_channel = self.task_mode == 'exclusive' and self.seg_classes > 1 and self.data_mode == 'multiclass'
        single_boundary = _aggregate_boundary_target(boundary, has_background_channel=has_background_channel)
        if self.disable_overlap_prior or self.disable_boundary_generators:
            shared_loss = seg_logits.new_zeros(())
            inner_loss = seg_logits.new_zeros(())
            disjoint_loss = seg_logits.new_zeros(())
        else:
            shared_loss = boundary_bce_dice_loss(preds['shared_boundary_prior'], single_boundary, self.eps, from_logits=False)
            inner_loss = boundary_bce_dice_loss(preds['inner_boundary_prior'], single_boundary, self.eps, from_logits=False)
            disjoint_loss = boundary_bce_dice_loss(preds['disjoint_boundary_prior'], single_boundary, self.eps, from_logits=False)

        if self.disable_overlap_prior or self.disable_region_to_edge_constraint or self.disable_boundary_generators:
            cons_loss = seg_logits.new_zeros(())
        else:
            cons_loss = self._consistency_loss(
                preds.get('edge_attention_intersection', preds['edge_attention']),
                preds.get('edge_attention_inner', preds['edge_attention']),
                preds.get('edge_attention_disjoint', None),
                preds['shared_boundary_prior'],
                preds['inner_boundary_prior'],
                preds.get('disjoint_boundary_prior', None),
            )

        lambda_aux = float(self.loss_cfg.get('lambda_aux', 0.3))
        lambda_aux_d2 = float(self.loss_cfg.get('lambda_aux_d2', 0.2))
        lambda_router = float(self.loss_cfg.get('lambda_router', 0.01))
        lambda_boundary_dist = float(self.loss_cfg.get('lambda_boundary_dist', 0.1))
        lambda_sdm = float(self.loss_cfg.get('lambda_sdm', 0.5))
        lambda_curvature = float(self.loss_cfg.get('lambda_curvature', 0.0))

        router_entropy = preds.get('router_entropy', seg_logits.new_zeros(()))

        # Boundary distance loss: penalizes predictions far from GT boundary
        if lambda_boundary_dist > 0:
            target_for_bdl = batch['mask'].float()
            if target_for_bdl.ndim == 3:
                target_for_bdl = target_for_bdl.unsqueeze(1)
            bdl = boundary_distance_loss(seg_logits, target_for_bdl)
        else:
            bdl = seg_logits.new_zeros(())

        # SDM supervision loss
        sdm_pred = preds.get('sdm_pred')
        if sdm_pred is not None and lambda_sdm > 0:
            target_for_sdm = batch['mask'].float()
            if target_for_sdm.ndim == 3:
                target_for_sdm = target_for_sdm.unsqueeze(1)
            sdm_l = sdm_loss(sdm_pred, target_for_sdm)
        else:
            sdm_l = seg_logits.new_zeros(())

        # Coarse SDM supervision (H/16 resolution, drives geometry-relation closed loop)
        lambda_coarse_sdm = float(self.loss_cfg.get('lambda_coarse_sdm', 0.0))
        coarse_sdm_pred = preds.get('coarse_sdm_pred')
        if coarse_sdm_pred is not None and lambda_coarse_sdm > 0:
            target_for_csdm = batch['mask'].float()
            if target_for_csdm.ndim == 3:
                target_for_csdm = target_for_csdm.unsqueeze(1)
            target_coarse = F.interpolate(target_for_csdm, size=coarse_sdm_pred.shape[-2:], mode='nearest')
            coarse_sdm_l = sdm_loss(coarse_sdm_pred, target_coarse)
        else:
            coarse_sdm_l = seg_logits.new_zeros(())

        # Boundary smoothness loss: penalizes jagged edges at GT boundary locations
        if lambda_curvature > 0:
            budgets_cfg = self.loss_cfg.get('curvature_budgets', [0.3] * seg_logits.shape[1])
            budgets = torch.tensor(budgets_cfg, dtype=torch.float32)
            target_for_curv = batch['mask'].float()
            if target_for_curv.ndim == 3:
                target_for_curv = target_for_curv.unsqueeze(1)
            curv_loss = boundary_smoothness_loss(seg_logits, target_for_curv, budgets)
        else:
            curv_loss = seg_logits.new_zeros(())

        # Gate regularization losses
        lambda_gate_entropy = float(self.loss_cfg.get('lambda_gate_entropy', 0.0))
        lambda_gate_diversity = float(self.loss_cfg.get('lambda_gate_diversity', 0.0))
        lambda_boundary_focal = float(self.loss_cfg.get('lambda_boundary_focal', 0.0))
        gate_acts = preds.get('gate_activations', [])
        if lambda_gate_entropy > 0 and gate_acts:
            g_entropy = gate_entropy_loss(gate_acts)
        else:
            g_entropy = seg_logits.new_zeros(())
        if lambda_gate_diversity > 0 and len(gate_acts) >= 2:
            g_diversity = gate_diversity_loss(gate_acts)
        else:
            g_diversity = seg_logits.new_zeros(())

        # Boundary focal loss: hard example mining on boundary pixels (with curvature weighting)
        if lambda_boundary_focal > 0:
            target_for_bfl = batch['mask'].float()
            if target_for_bfl.ndim == 3:
                target_for_bfl = target_for_bfl.unsqueeze(1)
            bfl_gamma = float(self.loss_cfg.get('boundary_focal_gamma', 2.0))
            bfl_width = int(self.loss_cfg.get('boundary_focal_width', 5))
            curv_w = None
            if sdm_pred is not None and self.seg_classes > 1:
                lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=sdm_pred.device)
                lap_k = lap_k.reshape(1, 1, 3, 3).repeat(sdm_pred.shape[1], 1, 1, 1)
                curv_raw = F.conv2d(sdm_pred, lap_k, padding=1, groups=sdm_pred.shape[1])
                curv_w = curv_raw.abs().mean(dim=1, keepdim=True)
                if curv_w.shape[-2:] != seg_logits.shape[-2:]:
                    curv_w = F.interpolate(curv_w, size=seg_logits.shape[-2:], mode='bilinear', align_corners=False)
            bfl = boundary_focal_loss(seg_logits, target_for_bfl, gamma=bfl_gamma, boundary_width=bfl_width, curvature_weight=curv_w)
        else:
            bfl = seg_logits.new_zeros(())

        # Direction field loss: supervise ICDC direction output with GT boundary normals
        lambda_direction = float(self.loss_cfg.get('lambda_direction', 0.0))
        direction_field = preds.get('direction_field')
        if lambda_direction > 0 and direction_field is not None:
            dir_loss = direction_field_loss(direction_field, batch['mask'], self.seg_classes)
        else:
            dir_loss = seg_logits.new_zeros(())

        # Topology-aware critical loss: penalize seg errors at topology-violating boundary regions
        lambda_topology = float(self.loss_cfg.get('lambda_topology', 0.0))
        relation_map = preds.get('relation_map')
        if lambda_topology > 0 and relation_map is not None and sdm_pred is not None and self.seg_classes > 1:
            target_for_topo = batch['mask'].float()
            if target_for_topo.ndim == 3:
                target_for_topo = target_for_topo.unsqueeze(1)
            topo_tau = float(self.loss_cfg.get('topology_tau', 2.0))
            topo_loss = topology_critical_loss(
                seg_logits, target_for_topo, relation_map, sdm_pred,
                relation_types=[], tau=topo_tau,
            )
        else:
            topo_loss = seg_logits.new_zeros(())

        # Adjacency boundary consistency: enforce complementarity at shared organ boundaries
        lambda_adjacency = float(self.loss_cfg.get('lambda_adjacency', 0.0))
        adj_pairs = self.loss_cfg.get('adjacency_pairs', [])
        if lambda_adjacency > 0 and sdm_pred is not None and adj_pairs and self.seg_classes > 1:
            adj_bw = float(self.loss_cfg.get('adjacency_boundary_width', 3.0))
            adj_loss = adjacency_boundary_loss(seg_logits, sdm_pred, adj_pairs, boundary_width=adj_bw)
        else:
            adj_loss = seg_logits.new_zeros(())

        # Cross-scale edge consistency loss
        lambda_edge_consistency = float(self.loss_cfg.get('lambda_edge_consistency', 0.0))
        if lambda_edge_consistency > 0 and not self.disable_edge_branch:
            edge_logits = preds['edge_logits']
            edge_cons = self._cross_scale_edge_consistency(edge_logits)
        else:
            edge_cons = seg_logits.new_zeros(())

        # Uncertainty calibration loss: uncertainty should correlate with prediction error
        lambda_uncertainty = float(self.loss_cfg.get('lambda_uncertainty', 0.0))
        uncertainty = preds.get('uncertainty')
        if lambda_uncertainty > 0 and uncertainty is not None:
            unc_loss = self._uncertainty_calibration_loss(seg_logits, batch['mask'], uncertainty)
        else:
            unc_loss = seg_logits.new_zeros(())

        # Uncertainty-guided segmentation: extra loss on uncertain regions
        lambda_unc_seg = float(self.loss_cfg.get('lambda_unc_seg', 0.0))
        if lambda_unc_seg > 0 and uncertainty is not None:
            target_unc = batch['mask'].float()
            if target_unc.ndim == 3:
                target_unc = target_unc.unsqueeze(1)
            unc_weight = uncertainty.detach()
            unc_bce = F.binary_cross_entropy_with_logits(seg_logits, target_unc, reduction='none')
            unc_seg_loss = (unc_bce * unc_weight).mean()
        else:
            unc_seg_loss = seg_logits.new_zeros(())

        total = (
            self.loss_cfg['lambda_seg'] * seg_loss
            + lambda_aux * aux_loss
            + lambda_aux_d2 * aux_loss_d2
            + self.loss_cfg['lambda_edge'] * edge_loss
            + self.loss_cfg['lambda_shared'] * shared_loss
            + self.loss_cfg['lambda_inner'] * inner_loss
            + self.loss_cfg.get('lambda_disjoint', 0.0) * disjoint_loss
            + self.loss_cfg['lambda_cons'] * cons_loss
            + lambda_router * router_entropy
            + lambda_boundary_dist * bdl
            + lambda_sdm * sdm_l
            + lambda_coarse_sdm * coarse_sdm_l
            + lambda_curvature * curv_loss
            + lambda_gate_entropy * g_entropy
            + lambda_gate_diversity * g_diversity
            + lambda_boundary_focal * bfl
            + lambda_direction * dir_loss
            + lambda_topology * topo_loss
            + lambda_adjacency * adj_loss
            + lambda_edge_consistency * edge_cons
            + lambda_uncertainty * unc_loss
            + lambda_unc_seg * unc_seg_loss
        )

        return {
            'loss': total,
            'seg_loss': seg_loss.detach(),
            'aux_loss': aux_loss.detach(),
            'dice_loss': seg_dice.detach(),
            'ce_bce_loss': seg_primary.detach(),
            'edge_loss': edge_loss.detach(),
            'shared_loss': shared_loss.detach(),
            'inner_loss': inner_loss.detach(),
            'disjoint_loss': disjoint_loss.detach(),
            'cons_loss': cons_loss.detach(),
            'boundary_dist_loss': bdl.detach(),
            'sdm_loss': sdm_l.detach(),
            'coarse_sdm_loss': coarse_sdm_l.detach(),
            'curvature_loss': curv_loss.detach(),
            'router_entropy': router_entropy.detach(),
            'gate_entropy_loss': g_entropy.detach(),
            'gate_diversity_loss': g_diversity.detach(),
            'boundary_focal_loss': bfl.detach(),
            'direction_loss': dir_loss.detach(),
            'topology_loss': topo_loss.detach(),
            'adjacency_loss': adj_loss.detach(),
            'edge_consistency_loss': edge_cons.detach(),
            'uncertainty_loss': unc_loss.detach(),
            'unc_seg_loss': unc_seg_loss.detach(),
        }
