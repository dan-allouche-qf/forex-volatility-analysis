import numpy as np
import pandas as pd
import pytest

from fxvol import backtest as bt


@pytest.fixture
def prices():
    rng = np.random.default_rng(3)
    return pd.Series(100 + rng.normal(0, 1, 400).cumsum(),
                     index=pd.bdate_range("2022-01-03", periods=400))


def test_no_look_ahead(prices):
    """Perturbing only the LAST price must not change any earlier strategy return."""
    sig = bt.sma_crossover_signal(prices, 20, 50)
    base = bt.backtest(prices, sig).returns

    bumped = prices.copy()
    bumped.iloc[-1] *= 1.10  # large shock to the final close only
    sig2 = bt.sma_crossover_signal(bumped, 20, 50)
    after = bt.backtest(bumped, sig2).returns

    # everything except the final day's return is identical
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1])


def test_costs_reduce_returns(prices):
    sig = bt.sma_crossover_signal(prices, 20, 50)
    free = bt.backtest(prices, sig, cost_bps=0.0).returns.sum()
    costly = bt.backtest(prices, sig, cost_bps=10.0).returns.sum()
    assert costly < free


def test_always_in_matches_buy_and_hold(prices):
    # Constant signal == 1, zero cost -> strategy reproduces the market exactly
    # (the lag is harmless because the signal never changes).
    sig = pd.Series(1.0, index=prices.index)
    res = bt.backtest(prices, sig, cost_bps=0.0)
    market = prices.pct_change().dropna()
    pd.testing.assert_series_equal(res.returns, market, check_names=False)


def test_performance_stats_keys(prices):
    sig = bt.sma_crossover_signal(prices, 20, 50)
    res = bt.backtest(prices, sig)
    for k in ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
              "turnover_annual", "hit_rate"):
        assert k in res.stats
