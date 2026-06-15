"""Range-based realized-variance estimators (Parkinson, Garman-Klass,
Rogers-Satchell).

The committed snapshot carries full OHLC but the close-only pipeline ignores
High/Low. Range estimators use the intraday range and are several times more
efficient than the squared-return proxy. Caveat: they assume no overnight gap,
no drift (Parkinson) and no jumps -- assumptions that bite in 24/5 FX daily bars,
so results are reported alongside the r^2 proxy, never as a silent replacement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import to_wide_ohlc

_LN2 = np.log(2.0)


def parkinson(high, low):
    """Parkinson (1980) daily variance from the high-low range."""
    return (np.log(high / low) ** 2) / (4.0 * _LN2)


def garman_klass(open_, high, low, close):
    """Garman-Klass (1980) daily variance from OHLC."""
    return 0.5 * np.log(high / low) ** 2 - (2.0 * _LN2 - 1.0) * np.log(close / open_) ** 2


def rogers_satchell(open_, high, low, close):
    """Rogers-Satchell (1991) daily variance (drift-robust)."""
    return (np.log(high / close) * np.log(high / open_)
            + np.log(low / close) * np.log(low / open_))


_ESTIMATORS = {"parkinson": "p", "garman_klass": "gk", "rogers_satchell": "rs"}


def realized_variance(long_ohlc: pd.DataFrame, pair: str, estimator: str = "parkinson") -> pd.Series:
    """Daily range-based variance (decimal return^2 units) for one pair."""
    wide = to_wide_ohlc(long_ohlc)
    o, h, low, c = (wide[f"{pair}_{x}"] for x in ("Open", "High", "Low", "Close"))
    if estimator == "parkinson":
        v = parkinson(h, low)
    elif estimator == "garman_klass":
        v = garman_klass(o, h, low, c)
    elif estimator == "rogers_satchell":
        v = rogers_satchell(o, h, low, c)
    else:  # pragma: no cover - guarded by tests
        raise ValueError(f"unknown estimator {estimator!r}; use {list(_ESTIMATORS)}")
    return v.dropna()


def realized_proxy(long_ohlc: pd.DataFrame, pair: str, estimator: str = "parkinson",
                   scale: float = 100.0) -> pd.Series:
    """Range-based realized variance rescaled to (``scale``*return)^2 units, so it
    is unit-compatible with :func:`fxvol.models.walk_forward_variance` forecasts
    (drop-in replacement for its ``realized`` column when ``scale=100``)."""
    return realized_variance(long_ohlc, pair, estimator) * (scale ** 2)
