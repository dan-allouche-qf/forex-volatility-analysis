import numpy as np
import pytest

from fxvol import data, realized


def test_parkinson_unbiased_on_simulated_paths():
    # Parkinson is unbiased for the variance of a zero-drift log-price over a bar.
    rng = np.random.default_rng(0)
    days, steps = 6000, 250
    true_var = 0.01 ** 2
    dt = true_var / steps
    highs, lows = np.empty(days), np.empty(days)
    for d in range(days):
        path = np.cumsum(rng.normal(0.0, np.sqrt(dt), steps))
        highs[d] = np.exp(path.max())
        lows[d] = np.exp(path.min())
    pk = realized.parkinson(highs, lows)
    assert (pk > 0).all()
    assert pk.mean() == pytest.approx(true_var, rel=0.10)


def test_estimators_nonnegative_on_real_ohlc():
    long = data.load_ohlc()
    for est in ("parkinson", "garman_klass", "rogers_satchell"):
        v = realized.realized_variance(long, "EURUSD", est)
        assert v.mean() > 0
        assert (v >= -1e-12).all()  # well-formed OHLC -> non-negative


def test_realized_proxy_units_match_squared_return_scale():
    long = data.load_ohlc()
    p = realized.realized_proxy(long, "EURUSD", "parkinson", scale=100.0)
    ann_vol_pct = np.sqrt(p.mean() * 252)  # annualized vol in %
    assert 2.0 < ann_vol_pct < 40.0


def test_unknown_estimator_raises():
    long = data.load_ohlc()
    with pytest.raises(ValueError):
        realized.realized_variance(long, "EURUSD", "bogus")
