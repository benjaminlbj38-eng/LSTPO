from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``base``.

    Parameters
    ----------
    base:
        Original configuration dictionary.
    updates:
        Override dictionary.

    Returns
    -------
    dict
        Merged dictionary without in-place modification of inputs.
    """
    base = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(paths: list[str | Path]) -> dict[str, Any]:
    """Load and merge multiple YAML config files from left to right."""
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_update(cfg, load_yaml(path))
    return cfg