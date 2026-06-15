"""Conditional-volatility models and an out-of-sample forecast horse race.

Baselines (EWMA, rolling, naive random-walk) vs GARCH(1,1) Student-t, scored on
1-day-ahead variance with QLIKE and MSE against the squared-return proxy, with a
Diebold-Mariano test for significance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

# arch fits more reliably on percent returns; we scale by 100 internally.
_SCALE = 100.0


@dataclass
class GarchFit:
    result: Any                  # arch ARCHModelResult (dynamic typing)
    cond_vol: pd.Series          # annualized conditional volatility, aligned to returns
    params: dict
    persistence: float
    long_run_vol: float          # annualized


def fit_garch(returns: pd.Series, dist: str = "t", p: int = 1, q: int = 1, o: int = 0,
              trading_days: int = 252) -> GarchFit:
    """Fit a GARCH(p,o,q) with the chosen innovation distribution (``o>0`` => GJR)."""
    r = returns.dropna() * _SCALE
    am = arch_model(r, mean="Constant", vol="GARCH", p=p, o=o, q=q, dist=dist)  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off")
    cond_vol = (res.conditional_volatility / _SCALE) * np.sqrt(trading_days)
    cond_vol.index = r.index
    pr = res.params
    alpha = pr.get("alpha[1]", 0.0)
    beta = pr.get("beta[1]", 0.0)
    gamma = pr.get("gamma[1]", 0.0)
    persistence = float(alpha + beta + o * gamma / 2.0)
    omega = pr.get("omega", np.nan)
    lr_var = omega / (1.0 - persistence) if persistence < 1 else np.nan  # in % units
    long_run_vol = float(np.sqrt(lr_var) / _SCALE * np.sqrt(trading_days))
    return GarchFit(
        result=res,
        cond_vol=cond_vol,
        params={k: float(v) for k, v in pr.items()},
        persistence=persistence,
        long_run_vol=long_run_vol,
    )


def standardized_residuals(fit: GarchFit) -> pd.Series:
    """Standardized residuals z = resid/conditional_vol of a fitted GARCH (unit
    variance under correct specification); the input to EVT tail modelling."""
    res = fit.result
    return (res.resid / res.conditional_volatility).dropna()


def walk_forward_variance(
    returns: pd.Series,
    split: float = 0.6,
    refit_every: int = 5,
    ewma_lambda: float = 0.94,
    roll_window: int = 21,
    dist: str = "t",
) -> pd.DataFrame:
    """One-day-ahead variance forecasts on an expanding window (daily units, % scale).

    Returns a DataFrame indexed by the forecast (t+1) date with columns:
    ``garch, ewma, rolling, naive`` (variance forecasts) and ``realized`` (the
    r_{t+1}^2 proxy). ``naive`` is the expanding unconditional (constant) variance.
    GARCH is refit every ``refit_every`` steps for speed.
    """
    r = returns.dropna() * _SCALE
    n = len(r)
    start = int(n * split)
    rv = r.values
    eps = 1e-8  # variance floor: a real forecaster never predicts exactly zero vol

    rows = {}
    last_fit: Any = None
    for t in range(start, n - 1):
        hist = r.iloc[: t + 1]
        # GARCH: refit periodically, forecast 1 step ahead
        if last_fit is None or (t - start) % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                last_fit = arch_model(hist, mean="Constant", vol="GARCH",
                                      p=1, q=1, dist=dist).fit(disp="off")  # type: ignore[arg-type]
        f = last_fit.forecast(horizon=1, reindex=False)
        garch_var = float(f.variance.values[-1, 0])

        # EWMA (RiskMetrics) recursion up to t -> forecast for t+1
        ewma_var = pd.Series(rv[: t + 1] ** 2).ewm(alpha=1 - ewma_lambda, adjust=False).mean().iloc[-1]
        rolling_var = float(np.var(rv[t + 1 - roll_window : t + 1], ddof=1))
        # "naive" = constant-variance benchmark: the expanding unconditional
        # variance to date. Tests whether modelling time-varying vol beats
        # assuming homoskedasticity at all.
        naive_var = float(np.var(rv[: t + 1], ddof=1))
        realized = float(rv[t + 1] ** 2)

        rows[r.index[t + 1]] = {
            "garch": max(garch_var, eps),
            "ewma": max(float(ewma_var), eps),
            "rolling": max(rolling_var, eps),
            "naive": max(naive_var, eps),
            "realized": realized,
        }
    return pd.DataFrame(rows).T


def walk_forward_garch(
    returns: pd.Series,
    refit_every: int = 10,
    dist: str = "t",
    min_train: int = 250,
    levels: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """Expanding-window 1-step-ahead GARCH(1,1) filter for conditional risk.

    For each out-of-sample date t+1 (using only information up to t) returns the
    conditional volatility forecast and the pieces needed for conditional VaR/ES:

    - ``sigma`` : 1-step conditional volatility, in DECIMAL return units
    - ``mu``    : fitted constant mean, decimal
    - ``nu``    : fitted Student-t degrees of freedom (NaN if ``dist!='t'``)
    - ``fhs_q_{L}`` : empirical lower-tail quantile (mass 1-L) of in-sample
      standardized residuals z = resid/sigma (for Filtered Historical Simulation)
    - ``fhs_es_{L}`` : ``-mean(z | z <= fhs_q_{L})`` (positive ES factor)

    GARCH is refit every ``refit_every`` steps; the 1-step forecast is recomputed
    every day. No look-ahead: row at t+1 uses only returns up to and including t.
    """
    r = returns.dropna() * _SCALE
    n = len(r)
    start = max(min_train, 1)
    rows: dict[Any, dict] = {}
    last_fit: Any = None
    mu: float = 0.0
    nu: float = float("nan")
    fhs: dict[float, tuple[float, float]] = {}
    for t in range(start, n - 1):
        if last_fit is None or (t - start) % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                last_fit = arch_model(r.iloc[: t + 1], mean="Constant", vol="GARCH",
                                      p=1, q=1, dist=dist).fit(disp="off")  # type: ignore[arg-type]
            pr = last_fit.params
            mu = float(pr.get("mu", 0.0))
            nu = float(pr.get("nu", np.nan))
            z = (last_fit.resid / last_fit.conditional_volatility).dropna().to_numpy()
            fhs = {}
            for lv in levels:
                q = float(np.quantile(z, 1.0 - lv))
                tail = z[z <= q]
                fhs[lv] = (q, float(-tail.mean()) if tail.size else float("nan"))
        f = last_fit.forecast(horizon=1, reindex=False)
        sigma_pct = float(np.sqrt(f.variance.values[-1, 0]))
        row = {"sigma": sigma_pct / _SCALE, "mu": mu / _SCALE, "nu": nu}
        for lv in levels:
            tag = int(lv * 100)
            row[f"fhs_q_{tag}"] = fhs[lv][0]
            row[f"fhs_es_{tag}"] = fhs[lv][1]
        rows[r.index[t + 1]] = row
    return pd.DataFrame(rows).T


def qlike(forecast_var: np.ndarray, realized_var: np.ndarray) -> np.ndarray:
    f = np.asarray(forecast_var, dtype=float)
    rz = np.asarray(realized_var, dtype=float)
    ratio = rz / f
    return ratio - np.log(ratio) - 1.0


def mse(forecast_var: np.ndarray, realized_var: np.ndarray) -> np.ndarray:
    return (np.asarray(realized_var, float) - np.asarray(forecast_var, float)) ** 2


def forecast_losses(wf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-observation QLIKE and MSE loss series per model.

    Observations with a zero realized proxy (degenerate flat days) are dropped so
    QLIKE stays finite; the same mask is applied to every model for fair
    comparison and for the Diebold-Mariano test.
    """
    models = [c for c in wf.columns if c != "realized"]
    valid = wf["realized"] > 0
    rz = wf.loc[valid, "realized"].values
    ql = pd.DataFrame({m: qlike(wf.loc[valid, m].values, rz) for m in models},
                      index=wf.index[valid])
    ms = pd.DataFrame({m: mse(wf.loc[valid, m].values, rz) for m in models},
                      index=wf.index[valid])
    return {"QLIKE": ql, "MSE": ms}


def score_forecasts(wf: pd.DataFrame) -> pd.DataFrame:
    """Mean QLIKE and MSE per model (lower is better)."""
    losses = forecast_losses(wf)
    out = {m: {"QLIKE": float(losses["QLIKE"][m].mean()),
               "MSE": float(losses["MSE"][m].mean())}
           for m in losses["QLIKE"].columns}
    return pd.DataFrame(out).T.sort_values("QLIKE")


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano test on two loss series (HAC variance, h-1 lags).

    Negative statistic => model A has lower loss (better) than B.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
        var += 2.0 * (1.0 - lag / h) * cov
    if var <= 0:  # constant loss differential -> degenerate variance
        var = np.finfo(float).eps
    dm = dbar / np.sqrt(var / n)
    return {"DM_stat": float(dm), "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(dm))))}
