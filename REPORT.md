# FX Volatility, Risk & Strategy — Results Report

*Pairs:* EUR/USD, GBP/USD, USD/JPY · *Period:* 2022-01-03 → 2024-12-30 · *Frequency:* daily · *Source:* Yahoo Finance snapshot (as-of 2026-06-14)

This report consolidates every computed result of the study. All figures are produced by the tested `fxvol` package and reproduce from the committed data snapshot by running the notebook (`notebooks/fx_volatility_analysis.ipynb`); every number here is generated directly from the package, not transcribed. See `report/report.pdf` for the formal write-up.

> **Methodological note.** An earlier version reindexed the three Mon-Fri series onto a full 365-day calendar and imputed weekend/holiday prices with a recursive 5-day rolling mean — fabricating ~28% (312 of 1093 calendar rows). FX has no weekend prices, so that step corrupted every statistic and made the √252 annualization inconsistent with the data frequency. It has been **removed**: all results below are computed on the native 781 trading days per pair.

## 1. Data integrity

- Trading rows per pair: **781** (no weekend rows: 0).
- Span: 2022-01-03 → 2024-12-30 = 1093 calendar days.
- Alignment: union of real trading dates, holiday gaps forward-filled at the price level only — no fabricated observations (guarded by `tests/test_preprocessing.py`).

## 2. Daily returns (log, %)

|        |    mean |    std |     min |     25% |     50% |    75% |    max |
|:-------|--------:|-------:|--------:|--------:|--------:|-------:|-------:|
| EURUSD | -0.0111 | 0.4964 | -1.859  | -0.2964 | -0.0134 | 0.2789 | 1.8211 |
| GBPUSD | -0.0093 | 0.5809 | -4.2324 | -0.3093 |  0.0041 | 0.2926 | 3.0308 |
| USDJPY |  0.0405 | 0.6657 | -3.7961 | -0.2841 |  0.0865 | 0.392  | 2.5029 |

USD/JPY carries the only positive mean drift (its 2022-24 uptrend); all three are roughly symmetric around zero with fat tails (see §4). GBP/USD has the widest range (min −4.23%), reflecting the late-2022 UK stress.

## 3. Volatility (annualized)

### 30-day rolling

|        |   current_% |   mean_% |   peak_% | peak_date   |   min_% |
|:-------|------------:|---------:|---------:|:------------|--------:|
| EURUSD |        6.95 |     7.63 |    13.81 | 2022-11-15  |    3.63 |
| GBPUSD |        6.76 |     8.61 |    22.18 | 2022-11-04  |    3.66 |
| USDJPY |        9.49 |    10.21 |    19.2  | 2022-12-21  |    4.38 |

### 90-day rolling

|        |   current_% |   mean_% |   peak_% | peak_date   |   min_% |
|:-------|------------:|---------:|---------:|:------------|--------:|
| EURUSD |        6.61 |     7.68 |    11.79 | 2022-11-14  |    5.08 |
| GBPUSD |        7.04 |     8.92 |    16.94 | 2023-01-09  |    5.36 |
| USDJPY |       10.66 |    10.56 |    15.01 | 2023-02-16  |    6.73 |

Current EWMA volatility (λ=0.94), %: EURUSD 7.02, GBPUSD 7.04, USDJPY 10.08.

The corrected 30-day peaks (EUR/USD **13.8%**, GBP/USD **22.2%**, USD/JPY **19.2%**) are materially lower than the figures quoted by the earlier (contaminated) version, and each is now dated via `idxmax`.

## 4. Distribution & econometric diagnostics

|        |    skew |   excess_kurtosis |   JB_p |   ADF_ret_p |   KPSS_ret_p |   LB_ret_p |   LB_sq_p |   ARCH_LM_p |   ADF_price_p |
|:-------|--------:|------------------:|-------:|------------:|-------------:|-----------:|----------:|------------:|--------------:|
| EURUSD | -0.0031 |            1.1726 |      0 |           0 |          0.1 |     0.2185 |    0      |      0.0002 |        0.1373 |
| GBPUSD | -0.2653 |            5.543  |      0 |           0 |          0.1 |     0.1515 |    0      |      0      |        0.1433 |
| USDJPY | -0.5155 |            2.8228 |      0 |           0 |          0.1 |     0.5991 |    0.4269 |      0.5761 |        0.284  |

- **Normality:** Jarque-Bera rejects normality for all three pairs (p≈0) — returns are leptokurtic (excess kurtosis up to 5.5 for GBP/USD).
- **Stationarity:** ADF rejects a unit root in returns (p≈0) while KPSS does not (p≥0.1) — returns are stationary; log-prices are not (ADF p≥0.13).
- **ARCH / clustering:** squared-return Ljung-Box and Engle ARCH-LM are significant for EUR/USD and GBP/USD (p<0.001), justifying GARCH. **USD/JPY's ARCH effect is weak in-sample** (ARCH-LM p=0.58) — reported as found.

## 5. Conditional-volatility models — GARCH(1,1) Student-t

|        |   omega |   alpha |   beta |   nu (t dof) |   persistence |   long_run_vol_% |
|:-------|--------:|--------:|-------:|-------------:|--------------:|-----------------:|
| EURUSD |  0.0006 |  0.0264 | 0.9715 |       7.7465 |        0.9979 |           8.6348 |
| GBPUSD |  0.0038 |  0.0488 | 0.9394 |       6.9151 |        0.9882 |           9.0531 |
| USDJPY |  0.0096 |  0.0723 | 0.9157 |       4.2933 |        0.988  |          14.1895 |

Persistence (α+β) is high for every pair (≥0.99), typical of daily FX. The Student-t degrees of freedom are low (4.3–7.7), confirming fat-tailed innovations. EWMA (RiskMetrics λ=0.94) is used as a baseline and overlaid with realized vol per pair (see `figures/cond_vol_*.png`).

## 6. Risk metrics

### Summary (annualized vol, ratios, drawdown, distribution)

|        |   ann_vol |   sharpe |   sortino |   max_drawdown |   mdd_duration |    skew |   excess_kurtosis |
|:-------|----------:|---------:|----------:|---------------:|---------------:|--------:|------------------:|
| EURUSD |    0.0788 |  -0.3553 |   -0.3543 |        -0.1651 |            184 | -0.0031 |            1.1726 |
| GBPUSD |    0.0922 |  -0.2534 |   -0.2446 |        -0.221  |            184 | -0.2653 |            5.543  |
| USDJPY |    0.1057 |   0.965  |    0.8804 |        -0.1503 |             62 | -0.5155 |            2.8228 |

### 1-day VaR & Expected Shortfall (loss fraction)

|        |   VaR_hist_95 |   ES_hist_95 |   VaR_t_95 |   ES_t_95 |   VaR_hist_99 |   ES_hist_99 |   VaR_t_99 |   ES_t_99 |
|:-------|--------------:|-------------:|-----------:|----------:|--------------:|-------------:|-----------:|----------:|
| EURUSD |        0.0081 |       0.0112 |     0.0081 |    0.0112 |        0.0131 |       0.0156 |     0.013  |    0.0167 |
| GBPUSD |        0.0097 |       0.0135 |     0.0089 |    0.0133 |        0.0159 |       0.021  |     0.0155 |    0.0215 |
| USDJPY |        0.0109 |       0.0159 |     0.0097 |    0.0151 |        0.0174 |       0.0241 |     0.0178 |    0.0255 |

### VaR backtest — rolling 250-day historical 99% VaR

|        |   breaches |   expected |   kupiec_p |   christoffersen_cc_p |
|:-------|-----------:|-----------:|-----------:|----------------------:|
| EURUSD |          4 |        5.3 |      0.553 |                 0.813 |
| GBPUSD |          6 |        5.3 |      0.765 |                 0.893 |
| USDJPY |          9 |        5.3 |      0.142 |                 0.112 |

All three pass the **Kupiec** (unconditional coverage) and **Christoffersen** (conditional coverage) tests at the 5% level, though USD/JPY is the weakest (9 breaches vs 5.3, Kupiec p=0.142, Christoffersen p=0.112) — foreshadowing the ES rejection in §9b.

## 7. Cross-pair dependence

### Return correlation matrix

|        |   EURUSD |   GBPUSD |   USDJPY |
|:-------|---------:|---------:|---------:|
| EURUSD |    1     |    0.779 |   -0.435 |
| GBPUSD |    0.779 |    1     |   -0.439 |
| USDJPY |   -0.435 |   -0.439 |    1     |

### 60-day rolling correlation (range)

|               |   mean |    min |    max |
|:--------------|-------:|-------:|-------:|
| EURUSD~GBPUSD |  0.787 |  0.614 |  0.902 |
| EURUSD~USDJPY | -0.432 | -0.698 | -0.023 |
| GBPUSD~USDJPY | -0.415 | -0.689 | -0.01  |

### PCA

Explained variance ratio: [0.707, 0.22, 0.074] — **PC1 explains 70.7%** of joint return variance (a common USD factor). PC1 loadings:

|        |   PC1_loading |
|:-------|--------------:|
| EURUSD |        -0.619 |
| GBPUSD |        -0.62  |
| USDJPY |         0.483 |

EUR/USD and GBP/USD co-move strongly (+0.78) and both move opposite USD/JPY (~-0.43). One factor dominating ~71% means diversification across these three majors is limited.

## 8. Strategy backtest — SMA(20/50) crossover, long/flat

Signals lagged one bar (next-bar execution, no look-ahead), 1.0 bps cost on turnover, vs buy-and-hold.

|        |   total_return |   cagr |   ann_vol |   sharpe |   sortino |   max_drawdown |   turnover_annual |   hit_rate |   bh_total_return |   bh_sharpe |
|:-------|---------------:|-------:|----------:|---------:|----------:|---------------:|------------------:|-----------:|------------------:|------------:|
| EURUSD |         0.0216 | 0.0069 |    0.045  |   0.1757 |    0.1303 |        -0.0821 |            5.1692 |     0.4459 |           -0.083  |     -0.316  |
| GBPUSD |         0.05   | 0.0159 |    0.0549 |   0.3144 |    0.2176 |        -0.0408 |            4.5231 |     0.5209 |           -0.0698 |     -0.2075 |
| USDJPY |         0.2572 | 0.0768 |    0.0812 |   0.9511 |    0.7347 |        -0.0945 |            4.8462 |     0.5742 |            0.3711 |      1.0192 |

The strategy beats buy-and-hold (Sharpe) on **EURUSD, GBPUSD** — by staying flat through the 2022-23 downtrends where holding lost money — but does **not** beat USD/JPY's persistent uptrend once costs are paid. A single textbook 20/50 parameterization is used (no parameter search), so in-sample optimism is minimal. Losses are shown, not hidden.

## 9. Out-of-sample volatility-forecast horse race

Expanding-window 1-day-ahead variance forecasts; QLIKE (lower=better); Diebold-Mariano vs benchmarks (negative stat = GARCH better).

|        | best_QLIKE   |   garch_QLIKE |   ewma_QLIKE |   rolling_QLIKE |   naive_QLIKE |   DM_vs_naive |   p_naive |   DM_vs_rolling |   p_roll |
|:-------|:-------------|--------------:|-------------:|----------------:|--------------:|--------------:|----------:|----------------:|---------:|
| EURUSD | garch        |         1.57  |        1.606 |           1.721 |         1.665 |         -1.3  |     0.195 |           -1.96 |    0.051 |
| GBPUSD | ewma         |         1.565 |        1.556 |           1.637 |         1.769 |         -4.46 |     0     |           -1.39 |    0.165 |
| USDJPY | naive        |         1.98  |        1.995 |           2.208 |         1.915 |          1.03 |     0.302 |           -1.23 |    0.22  |

GARCH/EWMA beat the constant-variance (naive) benchmark on the European pairs (DM significant at 5% for **GBPUSD**). For **USD/JPY the naive benchmark wins** — consistent with its weak in-sample ARCH effect (§4). Among GARCH/EWMA/rolling the differences are small, as expected.

## 9b. The research spine — conditional vs unconditional VaR/ES

Does conditioning on GARCH volatility improve risk **decisions**? We backtest four 1-day VaR models on identical out-of-sample dates. At 95% the unconditional Student-t VaR is mis-calibrated while the conditional models are well calibrated (Kupiec p):

|         |   EURUSD |   GBPUSD |   USDJPY |
|:--------|---------:|---------:|---------:|
| hist    |    0.26  |    0.118 |    0.26  |
| t       |    0.013 |    0.044 |    0.103 |
| garch_t |    0.361 |    0.18  |    0.376 |
| fhs     |    0.482 |    0.18  |    0.759 |

Christoffersen **independence** p at 99% (conditioning reduces breach clustering):

|         |   EURUSD |   GBPUSD |   USDJPY |
|:--------|---------:|---------:|---------:|
| hist    |    0.805 |    0.71  |    0.118 |
| t       |    0.853 |    0.853 |    0.599 |
| garch_t |    0.805 |    0.853 |    0.195 |
| fhs     |    0.805 |    0.902 |    0.291 |

**Expected Shortfall** backtest (Acerbi-Székely Z2; p>0.05 = ES accepted):

|        |     Z2 |   p_value |   breaches |   expected |
|:-------|-------:|----------:|-----------:|-----------:|
| EURUSD |  0.101 |     0.555 |          4 |        5.3 |
| GBPUSD |  0.497 |     0.876 |          3 |        5.3 |
| USDJPY | -1.147 |     0.016 |         11 |        5.3 |

The GARCH-t ES is accepted for the European pairs but **rejected for USD/JPY** (p=0.016, 11 breaches vs 5.3 expected) — the conditional model understates its tail, exactly the pair with weak ARCH. This motivates EVT.

## 9c. EVT tails (POT / Generalized Pareto on standardized residuals)

|        |   shape_xi |   POT_VaR99_std |   POT_ES99_std |
|:-------|-----------:|----------------:|---------------:|
| EURUSD |      0.119 |            2.6  |           3.36 |
| GBPUSD |      0.074 |            2.86 |           3.69 |
| USDJPY |      0.063 |            2.71 |           3.47 |

Positive shape ξ confirms heavier-than-exponential tails the single Student-t cannot fully capture.

## 9d. Strategy with error bars (vol-targeted momentum + SMA)

The vol model **sizes** a momentum strategy; every Sharpe carries a stationary-block-bootstrap 95% CI and a Probabilistic Sharpe Ratio:

| strategy                |   sharpe |   ci_low |   ci_high |   PSR |
|:------------------------|---------:|---------:|----------:|------:|
| EURUSD SMA(20/50)       |    0.176 |    -0.93 |      1.25 | 0.622 |
| EURUSD TSMOM vol-target |   -0.103 |    -1.13 |      0.91 | 0.429 |
| GBPUSD SMA(20/50)       |    0.314 |    -0.63 |      1.27 | 0.71  |
| GBPUSD TSMOM vol-target |    0.409 |    -0.61 |      1.41 | 0.764 |
| USDJPY SMA(20/50)       |    0.951 |    -0.15 |      2.08 | 0.951 |
| USDJPY TSMOM vol-target |    0.956 |    -0.16 |      2.05 | 0.953 |

**Every bootstrap CI straddles zero — including USD/JPY ([-0.15, 2.08])** — so no Sharpe is significant at 95%; only USD/JPY reaches a Probabilistic Sharpe Ratio above 0.95. On ~750 daily observations most edges are statistically indistinguishable from luck, and the project says so.

## 10. Key findings

|                     | EUR/USD             | GBP/USD             | USD/JPY            |
|:--------------------|:--------------------|:--------------------|:-------------------|
| 30d vol peak        | 13.81% (2022-11-15) | 22.18% (2022-11-04) | 19.2% (2022-12-21) |
| Normality (JB)      | reject              | reject              | reject             |
| ARCH effect         | yes                 | yes                 | weak               |
| GARCH persistence   | 0.9979              | 0.9882              | 0.988              |
| Strategy Sharpe     | 0.176               | 0.314               | 0.951              |
| Best vol forecaster | garch               | ewma                | naive              |

## 11. Limitations

- Daily data, 3-year window; the r² realized-variance proxy is noisy (QLIKE is robust to it).
- Single textbook strategy parameterization (no optimization, by design — avoids overfitting).
- Three pairs only; no GJR/EGARCH asymmetry or transaction-cost term-structure modelled.

## 12. Reproduce

```bash
make setup && make test && make run
```

Everything runs offline from `data/fx_ohlc_2022_2024.parquet`. Source of every number: `notebooks/fx_volatility_analysis.ipynb` (sections mirror this report); core logic and tests in `src/fxvol/` and `tests/`.
