"""Utility functions used by training and notebooks."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.config import MODELS_DIR, REPORT_DIR


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_project_dirs() -> None:
    """Create output directories used by scripts."""
    for directory in [MODELS_DIR, REPORT_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
