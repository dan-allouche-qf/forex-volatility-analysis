"""Statistical significance of backtests.

Turns point-estimate Sharpe ratios into inferential claims: a stationary block
bootstrap (Politis-Romano) for confidence intervals, the Probabilistic Sharpe
Ratio (Bailey & Lopez de Prado) accounting for skew/kurtosis and sample size, and
the Deflated Sharpe Ratio that discounts for the number of configurations tried.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_EULER = 0.5772156649015329


def _arr(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    return np.asarray(returns, dtype=float)


def stationary_bootstrap_indices(n: int, expected_block: float, n_boot: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap row indices, shape (n_boot, n).

    Block lengths are geometric with mean ``expected_block`` (probability
    ``p = 1/expected_block`` of starting a fresh block each step), preserving
    serial dependence in the resampled series.
    """
    p = 1.0 / expected_block
    idx = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        i = int(rng.integers(0, n))
        starts = rng.random(n) < p
        jumps = rng.integers(0, n, size=n)
        row = idx[b]
        for t in range(n):
            row[t] = i
            i = int(jumps[t]) if starts[t] else (i + 1) % n
    return idx


def sharpe_bootstrap(returns, trading_days: int = 252, expected_block: float = 10.0,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap distribution of the annualized Sharpe ratio.

    Returns the point estimate, a 95% percentile CI, and the bootstrap
    probability that the true Sharpe is <= 0.
    """
    r = _arr(returns)
    n = len(r)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * np.sqrt(trading_days)) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    samples = r[stationary_bootstrap_indices(n, expected_block, n_boot, rng)]
    mu = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    sr = np.where(sds > 0, mu / sds * np.sqrt(trading_days), np.nan)
    sr = sr[np.isfinite(sr)]
    return {
        "sharpe": point,
        "ci_low": float(np.percentile(sr, 2.5)),
        "ci_high": float(np.percentile(sr, 97.5)),
        "prob_sharpe_le_0": float(np.mean(sr <= 0.0)),
    }


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """PSR: probability the true (per-period) Sharpe exceeds ``sr_benchmark``,
    adjusting for non-normality (skew, kurtosis) and sample size."""
    r = _arr(returns)
    n = len(r)
    sd = r.std(ddof=1)
    if sd == 0 or n < 3:
        return float("nan")
    sr = r.mean() / sd
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))  # non-excess kurtosis
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(n - 1.0) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(var_trials_sr: float, n_trials: int) -> float:
    """Expected maximum (per-period) Sharpe across ``n_trials`` independent trials
    under the null of zero true Sharpe (Bailey & Lopez de Prado)."""
    e = np.e
    return float(np.sqrt(var_trials_sr) * (
        (1.0 - _EULER) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + _EULER * stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    ))


def deflated_sharpe_ratio(returns, n_trials: int, var_trials_sr: float) -> float:
    """DSR: PSR against the expected-maximum-Sharpe benchmark for ``n_trials``
    configurations, i.e. the probability the result survives selection bias.
    Requires ``n_trials >= 2``."""
    if n_trials < 2:
        raise ValueError("deflated_sharpe_ratio needs n_trials >= 2 (use PSR for one)")
    sr0 = expected_max_sharpe(var_trials_sr, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)
