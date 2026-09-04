#!/usr/bin/env python3
"""Entry point for CRABR ablation experiments.

Usage:
    python run_ablation.py                           # Run all ablations
    python run_ablation.py --quick                   # Quick debug run
    python run_ablation.py --configs configs/jsrt_scr.yaml
    python run_ablation.py --seeds 42 1337         # Custom seeds
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from ablation.runner import main

if __name__ == "__main__":
    main()
