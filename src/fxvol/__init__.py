"""fxvol - reproducible FX volatility, risk and backtesting toolkit.

The package is a thin, tested library layer; the notebook is a narrative on top.
Every analytical number in the project traces back to one of these functions.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (
    backtest,
    config,
    data,
    diagnostics,
    evt,
    indicators,
    models,
    plots,
    preprocessing,
    realized,
    risk,
    stats_backtest,
)

__all__ = [
    "config",
    "data",
    "preprocessing",
    "indicators",
    "realized",
    "risk",
    "models",
    "diagnostics",
    "stats_backtest",
    "evt",
    "backtest",
    "plots",
]
