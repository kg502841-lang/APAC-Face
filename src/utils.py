"""Shared helpers: configuration, paths, logging, seeding, device selection and JSON I/O."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        # stdout, not stderr: Windows PowerShell 5.1 turns native stderr lines into error records.
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S", stream=sys.stdout)
    return logging.getLogger(name)


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {cfg_path} is empty or malformed")
    return cfg


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive numpy/random seeds from the torch worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda":
        get_logger(__name__).warning("CUDA requested but not available; falling back to CPU")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)"
    return "CPU"


def auto_batch_size(requested: int, device: torch.device) -> int:
    """Cap the batch size by available VRAM (MobileNetV2 @224px with AMP uses ~60 MB/sample)."""
    if device.type != "cuda":
        return min(requested, 16)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if total_gb < 3:
        cap = 8
    elif total_gb < 5:
        cap = 16
    elif total_gb < 9:
        cap = 64
    else:
        cap = 128
    return max(1, min(requested, cap))


def save_json(obj: Any, path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return p


def load_json(path: str | os.PathLike) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
