import numpy as np
import pandas as pd
import pytest

from fxvol import indicators as ind


def test_log_returns_known_series():
    prices = pd.Series([100.0, 110.0, 99.0])
    lr = ind.log_returns(prices)
    assert lr.iloc[0] == pytest.approx(np.log(110 / 100))
    assert lr.iloc[1] == pytest.approx(np.log(99 / 110))


def test_simple_returns_known_series():
    prices = pd.Series([100.0, 110.0, 99.0])
    sr = ind.simple_returns(prices)
    assert sr.iloc[0] == pytest.approx(0.10)
    assert sr.iloc[1] == pytest.approx(-0.10)


def test_annualized_vol_identity():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 300))
    window = len(r)
    expected = r.std(ddof=1) * np.sqrt(252)
    got = ind.rolling_volatility(r, window, trading_days=252).iloc[-1]
    assert got == pytest.approx(expected)


def test_ewma_vol_matches_riskmetrics_recursion():
    r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
    lam = 0.94
    # manual recursion v_t = (1-lam) r_t^2 + lam v_{t-1}, seeded with r_0^2
    v = r.iloc[0] ** 2
    for x in r.iloc[1:]:
        v = (1 - lam) * x**2 + lam * v
    expected = np.sqrt(v * 252)
    got = ind.ewma_volatility(r, lam=lam, trading_days=252).iloc[-1]
    assert got == pytest.approx(expected)


def test_sma_known_series():
    prices = pd.Series([1.0, 2.0, 3.0, 4.0])
    s = ind.sma(prices, 2)
    assert s.iloc[1] == pytest.approx(1.5)
    assert s.iloc[3] == pytest.approx(3.5)


def test_rsi_cutler_golden():
    # Hand-computed Cutler RSI, window=2.
    prices = pd.Series([10.0, 11.0, 10.0, 12.0])
    r = ind.rsi(prices, window=2, method="cutler")
    assert r.iloc[2] == pytest.approx(50.0)
    assert r.iloc[3] == pytest.approx(100 - 100 / 3)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 40, dtype=float))
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    for method in ("wilder", "cutler"):
        ru = ind.rsi(up, 14, method).dropna()
        rd = ind.rsi(down, 14, method).dropna()
        assert (ru.between(0, 100)).all()
        assert ru.iloc[-1] == pytest.approx(100.0)  # only gains -> 100
        assert rd.iloc[-1] == pytest.approx(0.0)     # only losses -> 0


def test_rsi_wilder_differs_from_cutler():
    rng = np.random.default_rng(1)
    prices = pd.Series(100 + rng.normal(0, 1, 200).cumsum())
    w = ind.rsi(prices, 14, "wilder").dropna()
    c = ind.rsi(prices, 14, "cutler").dropna()
    assert not np.allclose(w.values, c.reindex(w.index).values)


def test_rsi_invalid_method():
    with pytest.raises(ValueError):
        ind.rsi(pd.Series([1.0, 2.0, 3.0]), method="nope")
