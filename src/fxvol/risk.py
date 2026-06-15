"""Risk metrics: VaR / Expected Shortfall, drawdown, Sharpe/Sortino, and VaR
backtests (Kupiec POF + Christoffersen).

Convention: returns are daily; VaR and ES are reported as POSITIVE loss numbers
at a confidence ``level`` (e.g. 0.95). ``alpha = 1 - level`` is the tail mass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- VaR / ES
def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    r = np.asarray(returns.dropna(), dtype=float)
    return float(-np.quantile(r, 1.0 - level))


def historical_es(returns: pd.Series, level: float = 0.95) -> float:
    r = np.asarray(returns.dropna(), dtype=float)
    q = np.quantile(r, 1.0 - level)
    tail = r[r <= q]
    return float(-tail.mean()) if tail.size else float("nan")


def normal_var(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    z = stats.norm.ppf(1.0 - level)
    return float(-(mu + sigma * z))


def normal_es(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    alpha = 1.0 - level
    z = stats.norm.ppf(alpha)
    return float(-(mu - sigma * stats.norm.pdf(z) / alpha))


def student_t_var(returns: pd.Series, level: float = 0.95) -> float:
    df, loc, scale = stats.t.fit(returns.dropna())
    q = stats.t.ppf(1.0 - level, df)
    return float(-(loc + scale * q))


def student_t_es(returns: pd.Series, level: float = 0.95) -> float:
    df, loc, scale = stats.t.fit(returns.dropna())
    alpha = 1.0 - level
    q = stats.t.ppf(alpha, df)
    # E[T | T <= q] for a standard t with `df` degrees of freedom (df > 1).
    e_tail = -((df + q**2) / (df - 1.0)) * stats.t.pdf(q, df) / alpha
    return float(-(loc + scale * e_tail))


# --------------------------------------------------------------------------- drawdown
def drawdown_curve(returns: pd.Series, log: bool = False) -> pd.Series:
    """Drawdown series (<= 0) from a return stream."""
    r = returns.dropna()
    equity = np.exp(r.cumsum()) if log else (1.0 + r).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series, log: bool = False) -> dict:
    """Max drawdown with peak/trough dates and duration (in observations)."""
    dd = drawdown_curve(returns, log=log)
    if dd.empty:
        return {"max_drawdown": float("nan"), "peak": None, "trough": None, "duration": 0}
    trough = dd.idxmin()
    pre = dd.loc[:trough]
    peak = pre[pre == 0.0].index.max() if (pre == 0.0).any() else dd.index[0]
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "duration": int(dd.loc[peak:trough].shape[0]),
    }


# --------------------------------------------------------------------------- ratios
def sharpe(returns: pd.Series, trading_days: int = 252, rf_annual: float = 0.0) -> float:
    r = returns.dropna()
    excess = r - rf_annual / trading_days
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(trading_days)) if sd > 0 else float("nan")


def sortino(returns: pd.Series, trading_days: int = 252, rf_annual: float = 0.0) -> float:
    r = returns.dropna()
    excess = r - rf_annual / trading_days
    downside = excess[excess < 0]
    # Downside deviation around the target (0), with Bessel correction so the
    # normalization matches Sharpe's std(ddof=1).
    dd = np.sqrt((downside**2).sum() / (downside.size - 1)) if downside.size > 1 else np.nan
    return float(excess.mean() / dd * np.sqrt(trading_days)) if dd and dd > 0 else float("nan")


# --------------------------------------------------------------------------- VaR backtests
def kupiec_pof(n: int, exceptions: int, level: float = 0.95) -> dict:
    """Kupiec unconditional-coverage (Proportion Of Failures) test."""
    p = 1.0 - level
    x = exceptions
    if x == 0:
        lr = -2.0 * (n * np.log(1.0 - p))
    else:
        pi = x / n
        lr = -2.0 * (
            (n - x) * np.log(1.0 - p) + x * np.log(p)
            - (n - x) * np.log(1.0 - pi) - x * np.log(pi)
        )
    return {
        "exceptions": int(x),
        "expected": float(n * p),
        "LR_pof": float(lr),
        "p_value": float(1.0 - stats.chi2.cdf(lr, 1)),
    }


def christoffersen(exceedances: pd.Series, level: float = 0.95) -> dict:
    """Christoffersen independence + conditional-coverage tests on the 0/1
    exception sequence."""
    h = np.asarray(exceedances, dtype=int)
    n = len(h)
    # transition counts n_ij : from state i to state j
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(h[:-1], h[1:], strict=False):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if n > 1 else 0.0

    def _safe(a, b):
        return a * np.log(b) if (a > 0 and b > 0) else 0.0

    ln_null = _safe(n00 + n10, 1 - pi) + _safe(n01 + n11, pi)
    ln_alt = (
        _safe(n00, 1 - pi01) + _safe(n01, pi01) + _safe(n10, 1 - pi11) + _safe(n11, pi11)
    )
    lr_ind = -2.0 * (ln_null - ln_alt)

    x = int(h.sum())
    lr_uc = kupiec_pof(n, x, level)["LR_pof"]
    lr_cc = lr_uc + lr_ind
    return {
        "LR_ind": float(lr_ind),
        "p_ind": float(1.0 - stats.chi2.cdf(lr_ind, 1)),
        "LR_cc": float(lr_cc),
        "p_cc": float(1.0 - stats.chi2.cdf(lr_cc, 2)),
    }


# --------------------------------------------------------------------------- table
def risk_table(
    returns: pd.DataFrame, levels=(0.95, 0.99), trading_days: int = 252
) -> pd.DataFrame:
    """Per-pair risk summary used as a headline deliverable."""
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        mdd = max_drawdown(r)
        row = {
            "ann_vol": float(r.std(ddof=1) * np.sqrt(trading_days)),
            "sharpe": sharpe(r, trading_days),
            "sortino": sortino(r, trading_days),
            "max_drawdown": mdd["max_drawdown"],
            "mdd_duration": mdd["duration"],
            "skew": float(r.skew()),
            "excess_kurtosis": float(r.kurt()),
        }
        for lv in levels:
            tag = f"{int(lv * 100)}"
            row[f"VaR_hist_{tag}"] = historical_var(r, lv)
            row[f"ES_hist_{tag}"] = historical_es(r, lv)
            row[f"VaR_t_{tag}"] = student_t_var(r, lv)
            row[f"ES_t_{tag}"] = student_t_es(r, lv)
        rows[col] = row
    return pd.DataFrame(rows).T


# ------------------------------------------------------- conditional (filtered) VaR/ES
def _std_t_q(alpha: float, nu: float) -> float:
    """Lower-tail quantile of a UNIT-VARIANCE standardized Student-t (as used by
    GARCH-t innovations), so it can be scaled directly by a conditional sigma."""
    return float(stats.t.ppf(alpha, nu) * np.sqrt((nu - 2.0) / nu))


def _std_t_es(alpha: float, nu: float) -> float:
    """Magnitude of the lower-tail mean of a unit-variance standardized Student-t."""
    q = stats.t.ppf(alpha, nu)
    es = ((nu + q**2) / (nu - 1.0)) * stats.t.pdf(q, nu) / alpha
    return float(es * np.sqrt((nu - 2.0) / nu))


def pinball_loss(returns, quantile_pred, alpha: float) -> float:
    """Quantile (pinball/tick) loss at level ``alpha`` for a predicted return
    quantile (``quantile_pred = -VaR``). Lower is better; proper for VaR."""
    r = np.asarray(returns, float)
    q = np.asarray(quantile_pred, float)
    d = r - q
    return float(np.mean(np.where(d >= 0, alpha * d, (alpha - 1.0) * d)))


def rolling_var_backtest(
    returns: pd.Series, garch_wf: pd.DataFrame, levels=(0.95, 0.99), window: int = 250
) -> pd.DataFrame:
    """Backtest UNCONDITIONAL (rolling historical, rolling Student-t) vs CONDITIONAL
    (GARCH-filtered Student-t, Filtered Historical Simulation) 1-day VaR on the same
    out-of-sample dates, scored by breaches, Kupiec, Christoffersen and pinball loss.

    ``garch_wf`` is the output of :func:`fxvol.models.walk_forward_garch`.
    """
    r = returns.dropna()
    dates = [d for d in garch_wf.index if r.index.get_loc(d) >= window]
    methods = ["hist", "t", "garch_t", "fhs"]
    acc: dict[tuple[str, float], list[float]] = {(m, lv): [] for m in methods for lv in levels}
    realized: list[float] = []
    for d in dates:
        pos = r.index.get_loc(d)
        win = r.iloc[pos - window: pos].to_numpy()
        df_, loc_, scale_ = stats.t.fit(win)
        g = garch_wf.loc[d]
        sigma, mu, nu = float(g["sigma"]), float(g["mu"]), float(g["nu"])
        for lv in levels:
            a = 1.0 - lv
            q = np.quantile(win, a)
            acc[("hist", lv)].append(-q)
            tq = stats.t.ppf(a, df_)
            acc[("t", lv)].append(-(loc_ + scale_ * tq))
            acc[("garch_t", lv)].append(-(mu + sigma * _std_t_q(a, nu)))
            acc[("fhs", lv)].append(-(mu + sigma * float(g[f"fhs_q_{int(lv * 100)}"])))
        realized.append(float(r.loc[d]))
    rz = np.asarray(realized)

    out = []
    for (m, lv), var_list in acc.items():
        var = np.asarray(var_list)
        exc = (rz < -var).astype(int)
        kp = kupiec_pof(len(exc), int(exc.sum()), lv)
        ch = christoffersen(pd.Series(exc), lv)
        out.append({
            "method": m, "level": lv, "breaches": int(exc.sum()),
            "expected": round(kp["expected"], 1),
            "kupiec_p": round(kp["p_value"], 3),
            "christ_ind_p": round(ch["p_ind"], 3),
            "christ_cc_p": round(ch["p_cc"], 3),
            "pinball": round(pinball_loss(rz, -var, 1.0 - lv), 6),
        })
    return pd.DataFrame(out).sort_values(["level", "method"]).reset_index(drop=True)


# ------------------------------------------------------------- ES backtest (Acerbi-Szekely)
def acerbi_szekely_z2(returns, var, es, level: float) -> float:
    """Acerbi-Szekely (2014) Test 2 statistic. E[Z2]=0 under correct ES;
    Z2 < 0 => ES underestimates tail risk."""
    r = np.asarray(returns, float)
    v = np.asarray(var, float)
    e = np.asarray(es, float)
    n = len(r)
    alpha = 1.0 - level
    ind = (r < -v).astype(float)
    return float(np.sum(r * ind / e) / (n * alpha) + 1.0)


def es_backtest_garch_t(
    returns: pd.Series, garch_wf: pd.DataFrame, level: float = 0.99,
    n_sim: int = 2000, seed: int = 0
) -> dict:
    """Acerbi-Szekely Z2 ES backtest for the GARCH-filtered Student-t model, with a
    simulated p-value from the model's own predictive distribution (left tail)."""
    r = returns.dropna().reindex(garch_wf.index).dropna()
    g = garch_wf.loc[r.index]
    sigma = g["sigma"].to_numpy()
    mu = g["mu"].to_numpy()
    nu = g["nu"].to_numpy()
    alpha = 1.0 - level
    var = -(mu + sigma * np.array([_std_t_q(alpha, n) for n in nu]))
    es = -mu + sigma * np.array([_std_t_es(alpha, n) for n in nu])
    rv = r.to_numpy()
    z2_obs = acerbi_szekely_z2(rv, var, es, level)

    rng = np.random.default_rng(seed)
    std = np.sqrt(nu / (nu - 2.0))
    sims = np.empty(n_sim)
    for i in range(n_sim):
        rsim = mu + sigma * (rng.standard_t(nu) / std)
        sims[i] = acerbi_szekely_z2(rsim, var, es, level)
    return {
        "Z2": z2_obs,
        "p_value": float(np.mean(sims <= z2_obs)),
        "breaches": int((rv < -var).sum()),
        "expected": round(len(rv) * alpha, 1),
    }


def vol_model_var_scorecard(
    returns: pd.Series, wf_var: pd.DataFrame, levels=(0.95, 0.99), nu: float | None = None
) -> pd.DataFrame:
    """Score each volatility model in a :func:`fxvol.models.walk_forward_variance`
    frame by the quality of the 1-day VaR it implies (breaches vs target,
    Christoffersen independence, pinball loss). Every model gets the SAME
    distributional assumption (Normal if ``nu`` is None, else standardized
    Student-t), so the comparison isolates the volatility forecast itself.
    """
    r = returns.dropna()
    model_cols = [c for c in wf_var.columns if c != "realized"]
    dates = [d for d in wf_var.index if d in r.index]
    realized = r.loc[dates].to_numpy()
    out = []
    for level in levels:
        a = 1.0 - level
        q = stats.norm.ppf(a) if nu is None else _std_t_q(a, nu)  # negative
        for m in model_cols:
            sigma = np.sqrt(wf_var.loc[dates, m].to_numpy()) / 100.0  # %-scale -> decimal
            var = -(sigma * q)
            exc = (realized < -var).astype(int)
            kp = kupiec_pof(len(exc), int(exc.sum()), level)
            ch = christoffersen(pd.Series(exc), level)
            out.append({
                "model": m, "level": level, "breaches": int(exc.sum()),
                "expected": round(kp["expected"], 1),
                "kupiec_p": round(kp["p_value"], 3),
                "christ_ind_p": round(ch["p_ind"], 3),
                "pinball": round(pinball_loss(realized, -var, a), 6),
            })
    return pd.DataFrame(out).sort_values(["level", "pinball"]).reset_index(drop=True)
