from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import evaluate_dataset, logits_to_prediction, metrics_for_batch, metrics_for_batch_per_class
from models import ERGASegmenter
from utils.config import load_config, save_json
from utils.dataset import build_dataset_from_config
from utils.visualization import (
    build_comparison_panel,
    build_overlay,
    build_palette,
    multilabel_to_color,
    palette_from_config,
    save_overlay,
    save_panel,
    save_prediction_mask,
)
from utils.tta import tta_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--head', type=str, default='', help='Run only a specific head (empty=all)')
    parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation')
    parser.add_argument('--tta-scales', type=str, default='0.9,1.0,1.1', help='TTA scale factors (comma-separated)')
    return parser.parse_args()


def loader_kwargs_from_cfg(cfg: dict) -> dict:
    kw = {
        'batch_size': cfg['train']['val_batch_size'],
        'shuffle': False,
        'num_workers': int(cfg['data'].get('num_workers', 0)),
        'pin_memory': bool(cfg['data'].get('pin_memory', False)),
    }
    if kw['num_workers'] > 0:
        kw['persistent_workers'] = bool(cfg['data'].get('persistent_workers', False))
        kw['prefetch_factor'] = int(cfg['data'].get('prefetch_factor', 2))
    return kw


def build_multi_head_palette(cfg: dict) -> dict[str, np.ndarray]:
    """Build per-head palettes from config."""
    infer_cfg = cfg.get('inference', {})
    colors = infer_cfg.get('class_colors', {})
    palettes = {}
    for head_cfg in cfg['model']['heads']:
        name = head_cfg['name']
        source_cfg = next(s for s in cfg['data']['sources'] if s['name'] == head_cfg['source'])
        class_names = source_cfg.get('class_names', [])
        head_colors = {k: v for k, v in colors.items() if k in class_names}
        palettes[name] = build_palette(head_colors, class_names=class_names, num_classes=head_cfg['num_classes'])
    return palettes


def build_unified_palette(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Build a single palette covering all heads' classes (concatenated)."""
    infer_cfg = cfg.get('inference', {})
    colors = infer_cfg.get('class_colors', {})
    all_names: list[str] = []
    for head_cfg in cfg['model']['heads']:
        source_cfg = next(s for s in cfg['data']['sources'] if s['name'] == head_cfg['source'])
        class_names = source_cfg.get('class_names', [])
        task_mode = head_cfg.get('task_mode', 'multilabel')
        if task_mode in ('exclusive', 'multiclass') and class_names and class_names[0].lower() == 'background':
            class_names = class_names[1:]
        all_names.extend(class_names)
    palette = build_palette(colors, class_names=all_names, num_classes=len(all_names))
    return palette, all_names


def infer_single_head(cfg: dict, args: argparse.Namespace) -> None:
    """Original single-head inference path."""
    checkpoint = args.checkpoint or cfg['train']['best_model_path']
    split = args.split
    prediction_dir = Path(cfg['inference']['prediction_dir'])
    prediction_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and cfg['train']['device'] == 'cuda' else 'cpu')
    dataset = build_dataset_from_config(cfg, split=split, is_train=False, include_raw_image=True)
    loader = DataLoader(dataset, **loader_kwargs_from_cfg(cfg))

    model = ERGASegmenter(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    if 'ema_model' in state:
        model.load_state_dict(state['ema_model'], strict=False)
    elif 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    use_tta = args.tta
    tta_scales = tuple(float(s) for s in args.tta_scales.split(','))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    if use_tta:
        print(f"TTA enabled: scales={tta_scales}, hflip=True")

    task_mode = cfg['model']['task_mode']
    num_classes = int(cfg['model']['num_classes'])
    class_names = cfg['data'].get('class_names', [])
    palette = palette_from_config(cfg)
    save_color_mask = cfg['inference'].get('save_color_mask', True)
    save_per_class = cfg['inference'].get('save_per_class', False)
    _mode_map = {'binary': 'exclusive', 'multiclass': 'exclusive', 'multilabel': 'independent'}
    normalized_mode = _mode_map.get(task_mode, task_mode)

    with torch.no_grad():
        for batch in tqdm(loader, desc='Infer', leave=False):
            image = batch['image'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and cfg['train']['amp'])):
                if use_tta:
                    logits = tta_predict(model, image, scales=tta_scales, flip_horizontal=True, flip_vertical=False)
                else:
                    logits = model(image)['seg_logits']
            pred = logits_to_prediction(logits, task_mode, cfg['data']['threshold']).cpu().numpy()
            raw_images = batch['raw_image'].numpy()
            gt_masks = batch['mask'].numpy()
            image_ids = batch['image_id']
            for idx in range(len(image_ids)):
                image_id = image_ids[idx]
                pred_mask = pred[idx]
                gt_mask = gt_masks[idx]
                save_prediction_mask(prediction_dir / f'{image_id}.png', pred_mask, task_mode, palette=palette, colorize=save_color_mask)
                if save_per_class and num_classes > 1:
                    sample_dir = prediction_dir / image_id
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    for ci in range(num_classes):
                        cn = class_names[ci] if ci < len(class_names) else f'class_{ci}'
                        cm = (pred_mask == ci).astype('uint8') if normalized_mode == 'exclusive' and pred_mask.ndim == 2 else pred_mask[ci]
                        save_prediction_mask(sample_dir / f'{cn}.png', cm, 'binary', palette=palette, colorize=save_color_mask, class_index=ci)
                if cfg['inference'].get('save_overlay', True):
                    overlay = build_overlay(raw_images[idx], pred_mask, cfg['inference']['overlay_alpha'], task_mode, palette)
                    save_overlay(prediction_dir / f'{image_id}_overlay.png', overlay)
                if cfg['inference'].get('save_panel', True):
                    panel = build_comparison_panel(raw_images[idx], pred_mask, task_mode, cfg['inference']['overlay_alpha'], ground_truth=gt_mask, palette=palette)
                    save_panel(prediction_dir / f'{image_id}_panel.png', panel)

    # Wrap model for TTA-aware evaluation
    if use_tta:
        class _TTAWrapper(torch.nn.Module):
            def __init__(self, base_model, scales):
                super().__init__()
                self.base_model = base_model
                self.scales = scales

            def forward(self, x):
                logits = tta_predict(self.base_model, x, scales=self.scales, flip_horizontal=True, flip_vertical=False)
                return {'seg_logits': logits}

        eval_model = _TTAWrapper(model, tta_scales)
    else:
        eval_model = model

    metrics = evaluate_dataset(eval_model, loader, cfg, device)

    # Per-class metrics
    class_names = cfg['data'].get('class_names', [])
    ignore_bg = cfg['data'].get('ignore_background', True)
    per_class_sums = {}
    per_class_counts = {}
    start_idx = 1 if (ignore_bg and num_classes > 1 and normalized_mode == 'exclusive') else 0
    effective_classes = num_classes - start_idx

    with torch.no_grad():
        for batch in tqdm(loader, desc='Per-class eval', leave=False):
            image = batch['image'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and cfg['train']['amp'])):
                if use_tta:
                    logits = tta_predict(model, image, scales=tta_scales, flip_horizontal=True, flip_vertical=False)
                else:
                    logits = model(image)['seg_logits']
            pred = logits_to_prediction(logits, task_mode, cfg['data']['threshold']).cpu().numpy()
            gt = batch['mask'].cpu().numpy()
            batch_pc = metrics_for_batch_per_class(pred, gt, task_mode, num_classes, ignore_background=ignore_bg)
            for sample_classes in batch_pc:
                for ci, cm in enumerate(sample_classes):
                    if ci not in per_class_sums:
                        per_class_sums[ci] = {k: 0.0 for k in cm}
                        per_class_counts[ci] = 0
                    for k in cm:
                        per_class_sums[ci][k] += cm[k]
                    per_class_counts[ci] += 1

    per_class_metrics = {}
    for ci in range(effective_classes):
        count = max(per_class_counts.get(ci, 0), 1)
        cn = class_names[ci + start_idx] if (ci + start_idx) < len(class_names) else f'class_{ci + start_idx}'
        per_class_metrics[cn] = {k: per_class_sums[ci][k] / count for k in per_class_sums.get(ci, {})}

    save_json(cfg['inference']['result_json'], {'overall': metrics, 'per_class': per_class_metrics})
    print('\n=== Overall ===')
    print({k: round(v, 5) for k, v in metrics.items()})
    print('\n=== Per-Class ===')
    for cn, cm in per_class_metrics.items():
        print(f"  {cn:20s}: dice={cm['dice']:.4f}  iou={cm['iou']:.4f}  f1={cm['f1']:.4f}  precision={cm['precision']:.4f}  recall={cm['recall']:.4f}  hd95={cm['hd95']:.2f}")


# PLACEHOLDER_MULTI_HEAD


def infer_multi_head(cfg: dict, args: argparse.Namespace) -> None:
    """Multi-head inference: unified output + per-group outputs."""
    checkpoint = args.checkpoint or cfg['train']['best_model_path']
    split = args.split
    prediction_dir = Path(cfg['inference']['prediction_dir'])
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / 'unified').mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and cfg['train']['device'] == 'cuda' else 'cpu')

    model = ERGASegmenter(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    if 'ema_model' in state:
        model.load_state_dict(state['ema_model'], strict=False)
    elif 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    head_cfgs = cfg['model']['heads']
    head_palettes = build_multi_head_palette(cfg)
    unified_palette, unified_names = build_unified_palette(cfg)
    threshold = float(cfg['data']['threshold'])
    alpha = cfg['inference']['overlay_alpha']
    save_color_mask = cfg['inference'].get('save_color_mask', True)
    target_heads = [h for h in head_cfgs if not args.head or h['name'] == args.head]

    for head_cfg in target_heads:
        (prediction_dir / head_cfg['name']).mkdir(parents=True, exist_ok=True)

    # Build per-source loaders to avoid collation issues (different mask shapes)
    from utils.dataset import build_dataset_from_source_cfg
    source_datasets = {}
    for src_cfg in cfg['data']['sources']:
        ds = build_dataset_from_source_cfg(src_cfg, cfg, split=split, is_train=False, include_raw_image=True)
        source_datasets[src_cfg['name']] = ds

    # Process each source independently
    with torch.no_grad():
        for head_cfg in target_heads:
            hname = head_cfg['name']
            source_name = head_cfg['source']
            if source_name not in source_datasets:
                continue
            ds = source_datasets[source_name]
            loader = DataLoader(ds, **loader_kwargs_from_cfg(cfg))
            task_mode = head_cfg.get('task_mode', 'multilabel')
            head_palette = head_palettes[hname]

            for batch in tqdm(loader, desc=f'Infer [{hname}]', leave=False):
                image = batch['image'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and cfg['train']['amp'])):
                    outputs = model(image)

                raw_images = batch['raw_image'].numpy()
                image_ids = batch['image_id']
                seg_logits = outputs[hname]['seg_logits']

                for idx in range(len(image_ids)):
                    image_id = image_ids[idx]
                    raw_img = raw_images[idx]
                    logit = seg_logits[idx]

                    if task_mode in ('exclusive', 'multiclass'):
                        pred_label = torch.softmax(logit, dim=0).argmax(dim=0).cpu().numpy()
                        save_prediction_mask(
                            prediction_dir / hname / f'{image_id}.png',
                            pred_label, task_mode, palette=head_palette, colorize=save_color_mask,
                        )
                        if cfg['inference'].get('save_overlay', True):
                            overlay = build_overlay(raw_img, pred_label, alpha, task_mode, head_palette)
                            save_overlay(prediction_dir / hname / f'{image_id}_overlay.png', overlay)
                    else:
                        pred_binary = (torch.sigmoid(logit) > threshold).cpu().numpy().astype(np.uint8)
                        save_prediction_mask(
                            prediction_dir / hname / f'{image_id}.png',
                            pred_binary, task_mode, palette=head_palette, colorize=save_color_mask,
                        )
                        if cfg['inference'].get('save_overlay', True):
                            overlay = build_overlay(raw_img, pred_binary, alpha, task_mode, head_palette)
                            save_overlay(prediction_dir / hname / f'{image_id}_overlay.png', overlay)

        # Unified output: run all heads on all sources
        print("\nGenerating unified predictions...")
        all_sources_ds = []
        for src_cfg in cfg['data']['sources']:
            all_sources_ds.append(source_datasets[src_cfg['name']])

        for ds in all_sources_ds:
            loader = DataLoader(ds, **loader_kwargs_from_cfg(cfg))
            for batch in tqdm(loader, desc='Unified', leave=False):
                image = batch['image'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and cfg['train']['amp'])):
                    outputs = model(image)

                raw_images = batch['raw_image'].numpy()
                image_ids = batch['image_id']

                for idx in range(len(image_ids)):
                    image_id = image_ids[idx]
                    raw_img = raw_images[idx]
                    unified_channels = []

                    for h_cfg in target_heads:
                        hn = h_cfg['name']
                        tm = h_cfg.get('task_mode', 'multilabel')
                        logit = outputs[hn]['seg_logits'][idx]
                        if tm in ('exclusive', 'multiclass'):
                            pred = torch.softmax(logit, dim=0).argmax(dim=0).cpu().numpy()
                            for ci in range(1, h_cfg['num_classes']):
                                unified_channels.append((pred == ci).astype(np.uint8))
                        else:
                            pred = (torch.sigmoid(logit) > threshold).cpu().numpy().astype(np.uint8)
                            for ci in range(pred.shape[0]):
                                unified_channels.append(pred[ci])

                    if unified_channels:
                        unified_mask = np.stack(unified_channels, axis=0)
                        save_prediction_mask(
                            prediction_dir / 'unified' / f'{image_id}.png',
                            unified_mask, 'multilabel', palette=unified_palette, colorize=save_color_mask,
                        )
                        if cfg['inference'].get('save_overlay', True):
                            overlay = build_overlay(raw_img, unified_mask, alpha, 'multilabel', unified_palette)
                            save_overlay(prediction_dir / 'unified' / f'{image_id}_overlay.png', overlay)

    print(f"\nResults saved to: {prediction_dir}")
    print(f"  - unified/: all classes merged ({len(unified_names)} channels)")
    for h in target_heads:
        print(f"  - {h['name']}/: {h['num_classes']} classes")

    # --- Per-head metrics evaluation ---
    print("\n=== Evaluating metrics per head ===")
    metric_keys = ['dice', 'iou', 'jaccard', 'precision', 'recall', 'f1', 'bf_score', 'hd95']
    all_head_metrics = {}
    all_head_per_class = {}

    for head_cfg in target_heads:
        hname = head_cfg['name']
        source_name = head_cfg['source']
        if source_name not in source_datasets:
            continue
        ds = source_datasets[source_name]
        loader = DataLoader(ds, **loader_kwargs_from_cfg(cfg))
        task_mode = head_cfg.get('task_mode', 'multilabel')
        num_classes = head_cfg['num_classes']
        source_cfg = next(s for s in cfg['data']['sources'] if s['name'] == source_name)
        class_names = source_cfg.get('class_names', [])
        ignore_bg = task_mode in ('exclusive', 'multiclass') and num_classes > 1

        effective_classes = num_classes - (1 if ignore_bg else 0)
        per_class_sums = [{k: 0.0 for k in metric_keys} for _ in range(effective_classes)]
        per_class_counts = [0] * effective_classes
        sums = {k: 0.0 for k in metric_keys}
        total_items = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Eval [{hname}]', leave=False):
                image = batch['image'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and cfg['train']['amp'])):
                    outputs = model(image)
                seg_logits = outputs[hname]['seg_logits']
                pred = logits_to_prediction(seg_logits, task_mode, threshold).cpu().numpy()
                gt = batch['mask'].cpu().numpy()

                batch_m = metrics_for_batch(pred, gt, task_mode, num_classes, ignore_background=ignore_bg)
                for item in batch_m:
                    for k in sums:
                        sums[k] += item[k]
                    total_items += 1

                batch_pc = metrics_for_batch_per_class(pred, gt, task_mode, num_classes, ignore_background=ignore_bg)
                for sample_classes in batch_pc:
                    for ci, cm in enumerate(sample_classes):
                        if ci < effective_classes:
                            for k in metric_keys:
                                per_class_sums[ci][k] += cm[k]
                            per_class_counts[ci] += 1

        total_items = max(total_items, 1)
        head_metrics = {k: v / total_items for k, v in sums.items()}
        all_head_metrics[hname] = head_metrics

        per_class_metrics = {}
        start_idx = 1 if ignore_bg else 0
        for ci in range(effective_classes):
            count = max(per_class_counts[ci], 1)
            cn = class_names[ci + start_idx] if (ci + start_idx) < len(class_names) else f'class_{ci + start_idx}'
            per_class_metrics[cn] = {k: per_class_sums[ci][k] / count for k, v in per_class_sums[ci].items()}
        all_head_per_class[hname] = per_class_metrics

        print(f"\n[{hname}] ({total_items} samples, {effective_classes} classes)")
        print(f"  Mean: {{{', '.join(f'{k}: {v:.5f}' for k, v in head_metrics.items())}}}")
        for cn, cm in per_class_metrics.items():
            print(f"  {cn:20s}: dice={cm['dice']:.4f}  iou={cm['iou']:.4f}  f1={cm['f1']:.4f}  hd95={cm['hd95']:.2f}")

    # Overall average
    if len(all_head_metrics) > 1:
        overall = {k: 0.0 for k in metric_keys}
        for hm in all_head_metrics.values():
            for k in overall:
                overall[k] += hm[k]
        overall = {k: v / len(all_head_metrics) for k, v in overall.items()}
        print(f"\n=== Overall (avg of {len(all_head_metrics)} heads) ===")
        print({k: round(v, 5) for k, v in overall.items()})
    else:
        overall = next(iter(all_head_metrics.values())) if all_head_metrics else {}

    # Save metrics JSON
    result_json = cfg['inference'].get('result_json', 'results/test_metrics_unified.json')
    Path(result_json).parent.mkdir(parents=True, exist_ok=True)
    save_json(result_json, {'per_head': all_head_metrics, 'per_class': all_head_per_class, 'overall': overall})
    print(f"\nMetrics saved to: {result_json}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    is_multi_head = bool(cfg.get('model', {}).get('multi_head', False))
    if is_multi_head:
        infer_multi_head(cfg, args)
    else:
        infer_single_head(cfg, args)


if __name__ == '__main__':
    main()
