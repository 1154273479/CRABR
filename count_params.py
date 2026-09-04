#!/usr/bin/env python3
"""
ERGA Model Parameter Calculator

This script calculates and displays the parameter count for the ERGA model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Disable CUDA to avoid potential issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch

from models import ERGASegmenter
from utils.config import load_config


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num: int) -> str:
    """Format large numbers with commas and appropriate units."""
    if num >= 1_000_000:
        return f"{num:,} ({num/1_000_000:.2f}M)"
    elif num >= 1_000:
        return f"{num:,} ({num/1_000:.1f}K)"
    else:
        return f"{num:,}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate ERGA model parameters")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config file {args.config} not found!")
        return

    try:
        cfg = load_config(args.config)
        
        # Create model on CPU
        device = torch.device('cpu')
        model = ERGASegmenter(cfg).to(device)
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        
        print("ERGA Model Parameter Analysis")
        print("=" * 50)
        print(f"Total parameters:     {format_number(total_params)}")
        print(f"Trainable parameters: {format_number(trainable_params)}")
        print(f"Non-trainable params: {format_number(total_params - trainable_params)}")
        
        # Calculate parameter efficiency
        if total_params > 0:
            trainable_ratio = trainable_params / total_params * 100
            print(f"Trainable ratio:      {trainable_ratio:.1f}%")
        
        # Print model configuration summary
        print("\nModel Configuration:")
        print("-" * 30)
        print(f"Task mode: {cfg['model']['task_mode']}")
        print(f"Number of classes: {cfg['model']['num_classes']}")
        default_region_mode = 'single_label' if cfg['model']['task_mode'] in {'binary', 'multiclass', 'exclusive'} else 'multi_label'
        print(f"Region mode: {cfg['model'].get('region_mode', default_region_mode)}")
        print(f"Region classes: {cfg['model']['region_num_classes']}")
        ablation = cfg['model'].get('ablation', {})
        print(f"Baseline: {ablation.get('baseline', 'erga')}")
        print(f"Cross-level fusion: {ablation.get('cross_level_fusion', ablation.get('cross_level_fusion_mode', 'gated'))}")
        print(f"Attention branches: {ablation.get('attention_branches', 'both')}")
        print(f"Relation guidance: {ablation.get('relation_guidance', 'full')}")
        print(f"ERGA fusion: {ablation.get('erga_fusion', True)}")
        print(f"Bottleneck enhancement: {ablation.get('bottleneck_enhancement', 'full')}")
        print(f"Embed dim: {cfg['model']['embed_dim']}")
        print(f"Common dim: {cfg['model']['common_dim']}")
        print(f"Decoder dim: {cfg['model']['decoder_dim']}")
        
        print("\nParameter calculation completed successfully!")
        
    except Exception as e:
        print(f"Error during parameter calculation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
