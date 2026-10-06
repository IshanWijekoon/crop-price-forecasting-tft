from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("paths", "forecast", "splits", "random_seed")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise KeyError(f"Config {path} is missing required keys: {missing}")

    split_total = sum(config["splits"].values())
    if abs(split_total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {split_total}")

    logger.debug("Loaded config from %s", path)
    return config


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
