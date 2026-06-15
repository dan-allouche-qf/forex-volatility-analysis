"""Vectorized SMA-crossover backtest with honest accounting.

No look-ahead: the position held on day ``t`` is decided from the signal observed
at the close of ``t-1`` (``signal.shift(1)``), i.e. next-bar execution.
Transaction costs are charged on turnover. A buy-and-hold benchmark is always
returned for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import simple_returns, sma
from .risk import max_drawdown, sharpe, sortino


def sma_crossover_signal(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long/flat target position (1/0) from an SMA crossover, decided at each close."""
    fast_sma = sma(prices, fast)
    slow_sma = sma(prices, slow)
    return (fast_sma > slow_sma).astype(float).where(slow_sma.notna())


def tsmom_signal(prices: pd.Series, lookback: int = 60) -> pd.Series:
    """Time-series-momentum signal: +1/-1 = sign of the trailing ``lookback``-day
    log return (Moskowitz-Ooi-Pedersen). Long/short, decided at each close."""
    trail = np.log(prices / prices.shift(lookback))
    return np.sign(trail).where(trail.notna())


def vol_target_position(signal: pd.Series, forecast_vol: pd.Series,
                        target_vol: float = 0.10, max_leverage: float = 3.0) -> pd.Series:
    """Scale a signal to a constant annualized volatility target using a
    (lagged-safe) conditional-vol forecast: ``pos = signal * target/forecast``,
    capped at ``max_leverage``. ``forecast_vol`` must use only info up to t."""
    lev = (target_vol / forecast_vol).clip(upper=max_leverage)
    return (signal * lev).where(signal.notna() & forecast_vol.notna())


@dataclass
class BacktestResult:
    returns: pd.Series       # strategy daily returns, net of costs
    equity: pd.Series        # strategy equity curve
    benchmark: pd.Series     # buy-and-hold equity curve
    position: pd.Series      # executed position (already lagged)
    stats: dict


def backtest(prices: pd.Series, signal: pd.Series, cost_bps: float = 1.0,
             trading_days: int = 252) -> BacktestResult:
    """Backtest a long/flat ``signal`` on ``prices`` with per-turnover costs."""
    rets = simple_returns(prices)
    # Execute next bar: position on day t uses the signal from t-1 -> no look-ahead.
    position = signal.shift(1).reindex(rets.index).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 1e4)
    strat = position * rets - cost

    equity = (1.0 + strat).cumprod()
    benchmark = (1.0 + rets).cumprod()
    stats = performance_stats(strat, trading_days)
    stats["turnover_annual"] = float(turnover.sum() / len(turnover) * trading_days)
    stats["bh_total_return"] = float(benchmark.iloc[-1] - 1.0)
    stats["bh_sharpe"] = sharpe(rets, trading_days)
    active = strat[position != 0]  # long OR short days (works for long/flat and long/short)
    stats["hit_rate"] = float((active > 0).mean()) if len(active) else float("nan")
    return BacktestResult(strat, equity, benchmark, position, stats)


def performance_stats(returns: pd.Series, trading_days: int = 252) -> dict:
    r = returns.dropna()
    n = len(r)
    total = float((1.0 + r).prod() - 1.0)
    cagr = float((1.0 + total) ** (trading_days / n) - 1.0) if n else float("nan")
    mdd = max_drawdown(r)
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(trading_days)),
        "sharpe": sharpe(r, trading_days),
        "sortino": sortino(r, trading_days),
        "max_drawdown": mdd["max_drawdown"],
    }
