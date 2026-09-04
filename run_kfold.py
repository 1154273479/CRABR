#!/usr/bin/env python3
"""Entry point for CRABR K-Fold Cross-Validation.

Usage:
    python run_kfold.py --config configs/jsrt_scr.yaml --folds 5 --seed 42
    python run_kfold.py --config configs/vindr_rib.yaml --gpu-ids 0,1
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from kfold.runner import main

if __name__ == "__main__":
    main()
