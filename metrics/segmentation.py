from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def binary_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    tp = float(((pred == 1) & (gt == 1)).sum())
    fp = float(((pred == 1) & (gt == 0)).sum())
    fn = float(((pred == 0) & (gt == 1)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    jaccard = iou
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {'dice': dice, 'iou': iou, 'jaccard': jaccard, 'precision': precision, 'recall': recall, 'f1': f1}


def boundary_map(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))


def bf_score(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    pred_boundary = boundary_map(pred)
    gt_boundary = boundary_map(gt)
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    pred_dilated = cv2.dilate(pred_boundary, kernel)
    gt_dilated = cv2.dilate(gt_boundary, kernel)
    precision = (pred_boundary * gt_dilated).sum() / max(pred_boundary.sum(), 1)
    recall = (gt_boundary * pred_dilated).sum() / max(gt_boundary.sum(), 1)
    return _safe_div(2.0 * precision * recall, precision + recall)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_boundary = boundary_map(pred)
    gt_boundary = boundary_map(gt)
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 0.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))
    dist_pred_to_gt = cv2.distanceTransform((1 - gt_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    dist_gt_to_pred = cv2.distanceTransform((1 - pred_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    distances = np.concatenate([dist_pred_to_gt[pred_boundary > 0], dist_gt_to_pred[gt_boundary > 0]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def logits_to_prediction(logits: torch.Tensor, task_mode: str, threshold: float) -> torch.Tensor:
    _map = {'binary': 'exclusive', 'multiclass': 'exclusive', 'multilabel': 'independent'}
    mode = _map.get(task_mode, task_mode)
    if mode == 'exclusive':
        if logits.shape[1] == 1:
            return (torch.sigmoid(logits) > threshold).float()
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)
    # independent
    return (torch.sigmoid(logits) > threshold).float()


def _binary_gt(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            return mask[:, 0].astype(np.uint8)
        return (mask[:, 1:] > 0.5).any(axis=1).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def multiclass_to_binary_masks(mask: np.ndarray, num_classes: int) -> List[np.ndarray]:
    return [(mask == class_idx).astype(np.uint8) for class_idx in range(num_classes)]


def multilabel_to_binary_masks(mask: np.ndarray) -> List[np.ndarray]:
    return [mask[class_idx].astype(np.uint8) for class_idx in range(mask.shape[0])]


def metrics_for_pair(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    stats = binary_stats(pred, gt)
    return {**stats, 'bf_score': bf_score(pred, gt), 'hd95': hd95(pred, gt)}


def metrics_for_batch(pred: np.ndarray, gt: np.ndarray, task_mode: str, num_classes: int, ignore_background: bool) -> List[Dict[str, float]]:
    _map = {'binary': 'exclusive', 'multiclass': 'exclusive', 'multilabel': 'independent'}
    mode = _map.get(task_mode, task_mode)
    batch_metrics = []
    for idx in range(pred.shape[0]):
        if mode == 'exclusive' and num_classes > 1:
            pred_channels = multiclass_to_binary_masks(pred[idx], num_classes)
            gt_channels = multiclass_to_binary_masks(gt[idx], num_classes)
            class_indices = range(1, num_classes) if ignore_background and num_classes > 1 else range(num_classes)
            per_class = []
            for class_idx in class_indices:
                if pred_channels[class_idx].sum() == 0 and gt_channels[class_idx].sum() == 0:
                    continue
                per_class.append(metrics_for_pair(pred_channels[class_idx], gt_channels[class_idx]))
        elif mode == 'independent':
            pred_channels = [pred[idx, class_idx].astype(np.uint8) for class_idx in range(pred.shape[1])]
            gt_channels = [gt[idx, class_idx].astype(np.uint8) for class_idx in range(gt.shape[1])]
            per_class = []
            for class_idx in range(len(pred_channels)):
                if pred_channels[class_idx].sum() == 0 and gt_channels[class_idx].sum() == 0:
                    continue
                per_class.append(metrics_for_pair(pred_channels[class_idx], gt_channels[class_idx]))
        else:
            pred_mask = pred[idx, 0].astype(np.uint8) if pred.ndim == 4 else pred[idx].astype(np.uint8)
            gt_mask = _binary_gt(gt[idx : idx + 1])[0]
            per_class = [metrics_for_pair(pred_mask, gt_mask)]

        if not per_class:
            continue
        batch_metrics.append({key: float(np.mean([m[key] for m in per_class])) for key in per_class[0].keys()})
    return batch_metrics


def metrics_for_batch_per_class(
    pred: np.ndarray, gt: np.ndarray, task_mode: str, num_classes: int, ignore_background: bool
) -> List[List[Dict[str, float]]]:
    """Return per-sample, per-class metrics. Each element is a list of num_classes dicts."""
    _map = {'binary': 'exclusive', 'multiclass': 'exclusive', 'multilabel': 'independent'}
    mode = _map.get(task_mode, task_mode)
    metric_keys = ['dice', 'iou', 'jaccard', 'precision', 'recall', 'f1', 'bf_score', 'hd95']
    batch_per_class: List[List[Dict[str, float]]] = []
    for idx in range(pred.shape[0]):
        if mode == 'exclusive' and num_classes > 1:
            pred_channels = multiclass_to_binary_masks(pred[idx], num_classes)
            gt_channels = multiclass_to_binary_masks(gt[idx], num_classes)
            start = 1 if ignore_background else 0
            sample_classes = []
            for class_idx in range(start, num_classes):
                if pred_channels[class_idx].sum() == 0 and gt_channels[class_idx].sum() == 0:
                    sample_classes.append({k: 1.0 for k in metric_keys})
                else:
                    sample_classes.append(metrics_for_pair(pred_channels[class_idx], gt_channels[class_idx]))
        elif mode == 'independent':
            pred_channels = [pred[idx, ci].astype(np.uint8) for ci in range(pred.shape[1])]
            gt_channels = [gt[idx, ci].astype(np.uint8) for ci in range(gt.shape[1])]
            sample_classes = []
            for ci in range(len(pred_channels)):
                if pred_channels[ci].sum() == 0 and gt_channels[ci].sum() == 0:
                    sample_classes.append({k: 1.0 for k in metric_keys})
                else:
                    sample_classes.append(metrics_for_pair(pred_channels[ci], gt_channels[ci]))
        else:
            pred_mask = pred[idx, 0].astype(np.uint8) if pred.ndim == 4 else pred[idx].astype(np.uint8)
            gt_mask = _binary_gt(gt[idx: idx + 1])[0]
            sample_classes = [metrics_for_pair(pred_mask, gt_mask)]
        batch_per_class.append(sample_classes)
    return batch_per_class


@torch.no_grad()
def evaluate_dataset(model: torch.nn.Module, loader, cfg: Dict, device: torch.device) -> Dict[str, float]:
    model.eval()
    metric_keys = ['dice', 'iou', 'jaccard', 'precision', 'recall', 'f1', 'bf_score', 'hd95']
    sums = {key: 0.0 for key in metric_keys}
    total_items = 0
    task_mode = cfg['model']['task_mode']
    num_classes = int(cfg['model']['num_classes'])

    for batch in tqdm(loader, desc='Eval', leave=False):
        image = batch['image'].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and cfg['train']['amp'])):
            outputs = model(image)
            logits = outputs['seg_logits'] if 'seg_logits' in outputs else outputs['logits']
        pred = logits_to_prediction(logits, task_mode, cfg['data']['threshold']).cpu().numpy()
        gt = batch['mask'].cpu().numpy()
        batch_metrics = metrics_for_batch(pred, gt, task_mode, num_classes, cfg['data'].get('ignore_background', True))
        for item in batch_metrics:
            for key in metric_keys:
                sums[key] += item[key]
            total_items += 1

    total_items = max(total_items, 1)
    return {key: value / total_items for key, value in sums.items()}


@torch.no_grad()
def evaluate_multi_head(model: torch.nn.Module, loader, cfg: Dict, device: torch.device) -> Dict[str, float]:
    """Evaluate multi-head model: compute metrics per head then average."""
    model.eval()
    metric_keys = ['dice', 'iou', 'jaccard', 'precision', 'recall', 'f1', 'bf_score', 'hd95']
    heads_cfg = cfg['model']['heads']
    threshold = float(cfg['data']['threshold'])

    per_head_sums: Dict[str, Dict[str, float]] = {}
    per_head_counts: Dict[str, int] = {}
    for h in heads_cfg:
        per_head_sums[h['name']] = {k: 0.0 for k in metric_keys}
        per_head_counts[h['name']] = 0

    for batch in tqdm(loader, desc='Eval', leave=False):
        image = batch['image'].to(device, non_blocking=True)
        source_ids = batch['source_id']
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and cfg['train']['amp'])):
            outputs = model(image)

        for head_idx, h_cfg in enumerate(heads_cfg):
            hname = h_cfg['name']
            if hname not in outputs:
                continue
            mask_sel = source_ids == head_idx
            if mask_sel.sum() == 0:
                continue

            seg_logits = outputs[hname]['seg_logits'][mask_sel]
            gt_mask = batch['mask'][mask_sel].cpu().numpy()
            task_mode = h_cfg.get('task_mode', 'multilabel')
            num_classes = h_cfg['num_classes']

            pred = logits_to_prediction(seg_logits, task_mode, threshold).cpu().numpy()
            batch_m = metrics_for_batch(pred, gt_mask, task_mode, num_classes, True)
            for item in batch_m:
                for k in metric_keys:
                    per_head_sums[hname][k] += item[k]
                per_head_counts[hname] += 1

    # Average across all heads
    overall = {k: 0.0 for k in metric_keys}
    active_heads = 0
    for hname in per_head_sums:
        count = max(per_head_counts[hname], 1)
        head_avg = {k: per_head_sums[hname][k] / count for k in metric_keys}
        if per_head_counts[hname] > 0:
            for k in metric_keys:
                overall[k] += head_avg[k]
            active_heads += 1

    active_heads = max(active_heads, 1)
    return {k: v / active_heads for k, v in overall.items()}
