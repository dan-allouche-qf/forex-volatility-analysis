"""Interactive FX volatility & risk dashboard.

Runs entirely off the committed parquet snapshot (no live data), so it deploys
cleanly to Streamlit Community Cloud and never breaks on a data feed.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

# Make the src/ package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxvol import backtest as bt  # noqa: E402
from fxvol import config, data, models, plots, risk  # noqa: E402
from fxvol import indicators as ind  # noqa: E402
from fxvol import preprocessing as pp  # noqa: E402
from fxvol import stats_backtest as sb  # noqa: E402

st.set_page_config(page_title="FX Volatility & Risk", layout="wide")
plots.apply_style()


@st.cache_data
def load():
    cfg = config.load_config()
    close = pp.align_close(data.load_ohlc())
    logret = ind.log_returns(close)
    return cfg, close, logret


cfg, close, logret = load()
td = cfg["indicators"]["trading_days"]
lam = cfg["models"]["ewma_lambda"]

st.title("FX Volatility, Risk & Strategy")
st.caption("Conditional volatility, backtested VaR/ES, and a vol-targeted strategy — "
           "computed live from the committed 2022–2024 snapshot via the tested `fxvol` package.")

pair = st.sidebar.selectbox("Currency pair", config.pairs())
level = st.sidebar.select_slider("VaR confidence", options=[0.95, 0.99], value=0.99)
st.sidebar.markdown("---")
st.sidebar.write("Source: Yahoo Finance snapshot, as-of " + cfg["data"]["as_of"])

r = logret[pair]

tab_vol, tab_risk, tab_strat = st.tabs(["Conditional volatility", "VaR backtest", "Strategy"])

with tab_vol:
    fit = models.fit_garch(r, dist=cfg["models"]["garch_dist"], trading_days=td)
    realized = ind.rolling_volatility(r, cfg["indicators"]["vol_windows"][0], td)
    ewma = ind.ewma_volatility(r, lam, td)
    c1, c2, c3 = st.columns(3)
    c1.metric("Current 30d vol", f"{realized.iloc[-1] * 100:.1f}%")
    c2.metric("GARCH persistence (α+β)", f"{fit.persistence:.3f}")
    c3.metric("Long-run vol", f"{fit.long_run_vol * 100:.1f}%")
    fig = plots.conditional_vol_overlay(pair, fit.cond_vol, realized, ewma)
    st.pyplot(fig)
    plt.close(fig)

with tab_risk:
    st.write(f"Rolling 250-day VaR backtest at {int(level * 100)}% — unconditional "
             "(historical, Student-t) vs conditional (GARCH-t, Filtered Historical Simulation).")
    wf_g = models.walk_forward_garch(r, refit_every=15, dist=cfg["models"]["garch_dist"],
                                     min_train=250, levels=(level,))
    sc = risk.rolling_var_backtest(r, wf_g, levels=(level,), window=250)
    st.dataframe(sc.set_index("method").round(3), use_container_width=True)
    st.caption("High Kupiec p = correct coverage; high Christoffersen independence p = no breach "
               "clustering. Conditioning typically improves independence in stress periods.")

with tab_strat:
    cost = cfg["backtest"]["cost_bps"]
    sma = bt.backtest(close[pair], bt.sma_crossover_signal(close[pair]), cost_bps=cost, trading_days=td)
    fvol = ind.ewma_volatility(r, lam, td)
    pos = bt.vol_target_position(bt.tsmom_signal(close[pair], cfg["backtest"]["tsmom_lookback"]),
                                 fvol, target_vol=cfg["backtest"]["target_vol"],
                                 max_leverage=cfg["backtest"]["max_leverage"])
    tsm = bt.backtest(close[pair], pos, cost_bps=cost, trading_days=td)
    for name, res in [("SMA(20/50)", sma), ("TSMOM vol-target", tsm)]:
        ci = sb.sharpe_bootstrap(res.returns, trading_days=td, n_boot=1000, seed=0)
        st.metric(f"{name} Sharpe", f"{ci['sharpe']:.2f}",
                  help=f"95% bootstrap CI [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
    fig = plots.equity_curve(sma, pair)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Sharpe shown with its bootstrap 95% CI: on ~750 daily observations most CIs straddle "
               "zero — the strategy edge is hard to distinguish from luck.")
