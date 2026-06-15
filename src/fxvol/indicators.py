"""Return, volatility and technical-indicator computations.

All functions operate on a trading-day-indexed price/return frame. Annualization
uses ``trading_days`` (== observations per year of the series), so the factor and
the data frequency are always consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log returns, ``ln(P_t / P_{t-1})``. Time-additive; the convention used
    throughout for volatility and aggregation."""
    return np.log(prices).diff().dropna()


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Arithmetic returns, ``P_t / P_{t-1} - 1`` (used for P&L compounding)."""
    return prices.pct_change().dropna()


def rolling_volatility(
    returns: pd.DataFrame | pd.Series, window: int, trading_days: int = 252
) -> pd.DataFrame | pd.Series:
    """Annualized rolling volatility = rolling std(returns) * sqrt(trading_days).

    ``window`` is counted in trading days because ``returns`` is a trading-day
    series, so a ``window`` of 30 is genuinely 30 trading days.
    """
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)


def ewma_volatility(
    returns: pd.DataFrame | pd.Series, lam: float = 0.94, trading_days: int = 252
) -> pd.DataFrame | pd.Series:
    """Annualized RiskMetrics EWMA volatility.

    Variance recursion ``v_t = (1-lam) * r_t^2 + lam * v_{t-1}`` (adjust=False),
    annualized to a volatility.
    """
    var = (returns**2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var * trading_days)


def sma(prices: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    """Simple moving average of prices over ``window`` trading days."""
    return prices.rolling(window).mean()


def rsi(prices: pd.Series, window: int = 14, method: str = "wilder") -> pd.Series:
    """Relative Strength Index.

    ``method="wilder"`` uses Wilder's recursive smoothing (EMA with
    ``alpha = 1/window``) - the canonical RSI. ``method="cutler"`` uses a simple
    moving average of gains/losses (Cutler's RSI). The two are different
    estimators; the choice is explicit rather than mislabeled.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    if method == "wilder":
        avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    elif method == "cutler":
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
    else:  # pragma: no cover - guarded by tests
        raise ValueError(f"unknown method={method!r}; use 'wilder' or 'cutler'")

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there are no losses in the window, RSI is 100 by definition.
    out = out.where(avg_loss != 0.0, 100.0)
    return out
