"""Econometric diagnostics: normality, stationarity, autocorrelation and ARCH
effects. Claims of "fat tails" / "volatility clustering" in the write-up are
substantiated by these tests rather than asserted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss


def jarque_bera(returns: pd.Series) -> dict:
    r = returns.dropna()
    jb, p = stats.jarque_bera(r)
    return {"JB": float(jb), "p_value": float(p), "skew": float(r.skew()),
            "excess_kurtosis": float(r.kurt())}


def adf_test(series: pd.Series) -> dict:
    stat, p, *_ = adfuller(series.dropna(), autolag="AIC")
    return {"ADF_stat": float(stat), "p_value": float(p)}


def kpss_test(series: pd.Series, regression: str = "c") -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when stat is outside the lookup table
        stat, p, *_ = kpss(series.dropna(), regression=regression, nlags="auto")
    return {"KPSS_stat": float(stat), "p_value": float(p)}


def ljung_box(series: pd.Series, lags: int = 10) -> dict:
    res = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    return {"LB_stat": float(res["lb_stat"].iloc[0]), "p_value": float(res["lb_pvalue"].iloc[0])}


def engle_arch(returns: pd.Series, lags: int = 10) -> dict:
    lm, lm_p, f, f_p = het_arch(returns.dropna(), nlags=lags)
    return {"LM_stat": float(lm), "p_value": float(lm_p)}


def diagnostics_table(returns: pd.DataFrame, prices: pd.DataFrame | None = None,
                      lags: int = 10) -> pd.DataFrame:
    """Per-pair diagnostics table.

    Columns: skew, excess kurtosis, Jarque-Bera (p), ADF on returns (p),
    KPSS on returns (p), Ljung-Box on returns and on squared returns (p),
    and Engle ARCH-LM (p). Low normality p-value => reject normality (fat tails);
    low ARCH-LM / squared-return Ljung-Box p-value => volatility clustering.
    """
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        jb = jarque_bera(r)
        row = {
            "skew": jb["skew"],
            "excess_kurtosis": jb["excess_kurtosis"],
            "JB_p": jb["p_value"],
            "ADF_ret_p": adf_test(r)["p_value"],
            "KPSS_ret_p": kpss_test(r)["p_value"],
            "LB_ret_p": ljung_box(r, lags)["p_value"],
            "LB_sq_p": ljung_box(r**2, lags)["p_value"],
            "ARCH_LM_p": engle_arch(r, lags)["p_value"],
        }
        if prices is not None and col in prices.columns:
            row["ADF_price_p"] = adf_test(np.log(prices[col]))["p_value"]
        rows[col] = row
    return pd.DataFrame(rows).T
