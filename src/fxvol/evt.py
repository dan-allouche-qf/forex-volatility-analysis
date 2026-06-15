"""Extreme Value Theory tails: Peaks-Over-Threshold (Generalized Pareto) on
GARCH-standardized residuals (McNeil-Frey: GARCH for the body dynamics, EVT for
the tail). A single fitted Student-t imposes a symmetric polynomial tail; EVT lets
the data choose the tail index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def gpd_fit(losses, threshold_quantile: float = 0.90) -> dict:
    """Fit a Generalized Pareto Distribution to exceedances of ``losses`` over a
    high threshold (loc fixed at 0). ``losses`` are positive tail values
    (e.g. ``-standardized_residual``)."""
    x = np.asarray(losses, float)
    x = x[np.isfinite(x)]
    n = len(x)
    u = float(np.quantile(x, threshold_quantile))
    exc = x[x > u] - u
    shape, _loc, scale = stats.genpareto.fit(exc, floc=0.0)
    return {"shape": float(shape), "scale": float(scale), "threshold": u,
            "n": n, "n_exceed": int(exc.size)}


def pot_var_es(losses, level: float = 0.99, threshold_quantile: float = 0.90) -> dict:
    """POT VaR/ES at confidence ``level`` on the scale of ``losses`` (standardized).

    Uses the GPD tail estimator F-bar(x) = (Nu/n)(1 + shape*(x-u)/scale)^(-1/shape).
    """
    f = gpd_fit(losses, threshold_quantile)
    c, scale, u, n, nu = f["shape"], f["scale"], f["threshold"], f["n"], f["n_exceed"]
    p = 1.0 - level
    ratio = p * n / nu
    if abs(c) < 1e-6:  # exponential limit (shape -> 0)
        var = u + scale * (-np.log(ratio))
    else:
        var = u + (scale / c) * (ratio ** (-c) - 1.0)
    es = (var + scale - c * u) / (1.0 - c) if c < 1.0 else float("nan")
    return {**f, "level": level, "VaR": float(var), "ES": float(es)}


def mean_excess(losses, n_points: int = 40) -> pd.DataFrame:
    """Mean-excess function e(u) = E[X-u | X>u] over a grid of thresholds, used to
    justify the POT threshold (linear-in-u region indicates GPD applicability)."""
    x = np.asarray(losses, float)
    x = x[np.isfinite(x)]
    lo, hi = np.quantile(x, 0.50), np.quantile(x, 0.98)
    us = np.linspace(lo, hi, n_points)
    rows = []
    for u in us:
        exc = x[x > u] - u
        if exc.size >= 5:
            rows.append({"threshold": float(u), "mean_excess": float(exc.mean()),
                         "n_exceed": int(exc.size)})
    return pd.DataFrame(rows)


def conditional_pot_var_es(sigma_next: float, std_resid_losses, level: float = 0.99,
                           threshold_quantile: float = 0.90) -> dict:
    """McNeil-Frey conditional VaR/ES: POT tail of standardized-residual losses,
    rescaled by the 1-step conditional volatility ``sigma_next``."""
    pot = pot_var_es(std_resid_losses, level, threshold_quantile)
    return {**pot, "cond_VaR": float(sigma_next * pot["VaR"]),
            "cond_ES": float(sigma_next * pot["ES"])}
