"""Canonical recomputation of the project's headline numbers from the package.

Single source of truth for the figures quoted in README / REPORT.md / report.tex.
Used by ``scripts/check_numbers.py`` (the provenance guard) so committed prose can
never silently drift from what the code produces.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from fxvol import backtest as bt  # noqa: E402
from fxvol import config, data, evt, models, risk  # noqa: E402
from fxvol import indicators as ind  # noqa: E402
from fxvol import preprocessing as pp  # noqa: E402
from fxvol import stats_backtest as sb  # noqa: E402


def compute_key() -> dict:
    """Recompute the headline KEY numbers (mirrors the notebook's KEY dict)."""
    cfg = config.load_config()
    td = cfg["indicators"]["trading_days"]
    pairs = config.pairs()
    levels = tuple(cfg["risk"]["var_levels"])
    lam = cfg["models"]["ewma_lambda"]
    short = cfg["indicators"]["vol_windows"][0]

    close = pp.align_close(data.load_ohlc())
    logret = ind.log_returns(close)
    vol30 = ind.rolling_volatility(logret, short, td)

    key: dict = {
        "vol_peak_%": {p: round(float(vol30[p].max() * 100), 2) for p in pairs},
        "vol_peak_date": {p: str(vol30[p].idxmax().date()) for p in pairs},
        "vol_current_%": {p: round(float(vol30[p].iloc[-1] * 100), 2) for p in pairs},
    }

    corr = logret.corr()
    ev = np.linalg.eigvalsh(corr.values)
    key["pc1_explained_%"] = round(float(ev[::-1][0] / ev.sum() * 100), 1)

    garch = {p: models.fit_garch(logret[p], dist=cfg["models"]["garch_dist"], trading_days=td)
             for p in pairs}
    key["garch_persistence"] = {p: round(garch[p].persistence, 4) for p in pairs}
    key["evt_shape_xi"] = {}
    for p in pairs:
        z = models.standardized_residuals(garch[p])
        pot = evt.pot_var_es((-z).to_numpy(), level=0.99,
                             threshold_quantile=cfg["evt"]["threshold_quantile"])
        key["evt_shape_xi"][p] = round(pot["shape"], 3)

    # conditional VaR/ES spine + ES backtest
    key["es99_garch_t_pvalue"] = {}
    for p in pairs:
        wf_g = models.walk_forward_garch(logret[p], refit_every=15,
                                         dist=cfg["models"]["garch_dist"], min_train=250, levels=levels)
        es = risk.es_backtest_garch_t(logret[p], wf_g, level=0.99, n_sim=2000, seed=0)
        key["es99_garch_t_pvalue"][p] = round(es["p_value"], 3)

    # strategy Sharpe + significance
    cost = cfg["backtest"]["cost_bps"]
    sma, tsm = {}, {}
    for p in pairs:
        sma[p] = bt.backtest(close[p], bt.sma_crossover_signal(close[p]), cost_bps=cost, trading_days=td)
        fvol = ind.ewma_volatility(logret[p], lam, td)
        pos = bt.vol_target_position(bt.tsmom_signal(close[p], cfg["backtest"]["tsmom_lookback"]),
                                     fvol, target_vol=cfg["backtest"]["target_vol"],
                                     max_leverage=cfg["backtest"]["max_leverage"])
        tsm[p] = bt.backtest(close[p], pos, cost_bps=cost, trading_days=td)
    key["strategy_sharpe"] = {p: round(sma[p].stats["sharpe"], 3) for p in pairs}
    sig = []
    for p in pairs:
        for name, r in [("SMA", sma[p].returns), ("TSMOM", tsm[p].returns)]:
            if sb.probabilistic_sharpe_ratio(r.dropna()) > 0.95:
                sig.append(f"{p}:{name}")
    key["sharpe_significant_psr95"] = sorted(sig)
    return key


if __name__ == "__main__":
    import json
    print(json.dumps(compute_key(), indent=2))
