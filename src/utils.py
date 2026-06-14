"""Utilities: seeding, JSON logging, and (optional) MP4 recording helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Mapping[str, Any], path: str | os.PathLike) -> None:
    """Save a dict as JSON. NumPy arrays are converted to lists."""
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Type {type(o)} not serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return json.load(f)
