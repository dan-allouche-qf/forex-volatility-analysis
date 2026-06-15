# FX Volatility, Risk & Strategy

A reproducible quantitative study of **EUR/USD, GBP/USD, USD/JPY** (daily, 2022–2024), organised around
one research question:

!!! question "Research question"
    **Does conditioning FX risk on time-varying, fat-tailed volatility actually improve real risk
    *decisions* — VaR/ES coverage and exception independence — or is a parsimonious constant-variance
    model good enough?**

## The apparatus

1. **Diagnostics** — Jarque–Bera, ADF/KPSS, Ljung–Box, Engle ARCH-LM establish fat tails and clustering.
2. **Conditional volatility** — EWMA and GARCH(1,1) Student-t.
3. **Conditional risk** — GARCH-filtered and Filtered-Historical-Simulation VaR/ES, **backtested
   side-by-side against the unconditional model** (Kupiec, Christoffersen, pinball), with an
   Acerbi–Székely **ES** backtest.
4. **Extreme value theory** — Peaks-Over-Threshold (Generalized Pareto) tails on standardized residuals.
5. **Strategy** — a volatility-targeted time-series-momentum strategy, with **bootstrap Sharpe confidence
   intervals** and the Probabilistic Sharpe Ratio.
6. **Forecasting** — an out-of-sample volatility race scored by QLIKE and by risk-decision quality.

## Honest headline

Conditioning improves VaR exception *independence* for the European pairs; but the GARCH-t Expected
Shortfall is **rejected for USD/JPY** (Acerbi–Székely *p* = 0.016), which motivates the EVT tail layer.
And on ~750 daily observations, every strategy Sharpe confidence interval straddles zero except USD/JPY —
most "edges" are statistically indistinguishable from luck, and the project says so.

## Reproduce

```bash
make setup        # venv + install
make test         # 60 unit tests (golden values, no-look-ahead, calibration)
make run          # execute the notebook offline from the snapshot
make report-guard # fail if any committed number drifts from the computed value
streamlit run app/streamlit_app.py   # interactive dashboard
```

Everything runs offline from `data/fx_ohlc_2022_2024.parquet`. See the
[compiled PDF report](https://github.com/dan-allouche-qf/forex-volatility-analysis/blob/main/report/report.pdf)
for the full write-up, and the [API reference](api.md) for the package.
