"""Configuration loading. ``config.yaml`` is the single source of truth."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root (the directory that contains ``config.yaml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").exists():
            return parent
    # Fallback: two levels up from src/fxvol/
    return here.parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache ``config.yaml`` as a nested dict."""
    cfg_path = Path(path) if path is not None else project_root() / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    relative = Path(relative)
    return relative if relative.is_absolute() else project_root() / relative


def pairs() -> list[str]:
    """Ordered list of currency-pair names (e.g. ['EURUSD', 'GBPUSD', 'USDJPY'])."""
    return list(load_config()["data"]["tickers"].keys())
