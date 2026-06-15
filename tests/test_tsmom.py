import numpy as np
import pandas as pd
import pytest

from fxvol import backtest as bt
from fxvol import indicators as ind


@pytest.fixture
def prices():
    rng = np.random.default_rng(5)
    return pd.Series(100 + rng.normal(0.05, 1.0, 600).cumsum(),
                     index=pd.bdate_range("2021-01-04", periods=600))


def test_tsmom_signal_is_sign_of_trailing_return(prices):
    sig = bt.tsmom_signal(prices, lookback=60).dropna()
    assert set(np.unique(sig.values)).issubset({-1.0, 0.0, 1.0})
    # manual check on one date
    d = sig.index[100]
    pos = prices.index.get_loc(d)
    trail = np.log(prices.iloc[pos] / prices.iloc[pos - 60])
    assert sig.loc[d] == np.sign(trail)


def test_vol_target_scales_inversely_with_forecast_vol(prices):
    sig = pd.Series(1.0, index=prices.index)
    lo_vol = pd.Series(0.05, index=prices.index)
    hi_vol = pd.Series(0.20, index=prices.index)
    pos_lo = bt.vol_target_position(sig, lo_vol, target_vol=0.10, max_leverage=10).dropna()
    pos_hi = bt.vol_target_position(sig, hi_vol, target_vol=0.10, max_leverage=10).dropna()
    assert (pos_lo > pos_hi).all()
    assert pos_lo.iloc[0] == pytest.approx(2.0)   # 0.10/0.05
    assert pos_hi.iloc[0] == pytest.approx(0.5)   # 0.10/0.20


def test_vol_target_respects_leverage_cap(prices):
    sig = pd.Series(1.0, index=prices.index)
    tiny_vol = pd.Series(0.01, index=prices.index)
    pos = bt.vol_target_position(sig, tiny_vol, target_vol=0.10, max_leverage=3.0).dropna()
    assert (pos <= 3.0 + 1e-9).all()


def test_tsmom_vol_target_no_look_ahead(prices):
    rets = ind.log_returns(prices)
    fvol = ind.ewma_volatility(rets, lam=0.94)
    sig = bt.tsmom_signal(prices, lookback=60)
    pos = bt.vol_target_position(sig, fvol, target_vol=0.10)
    base = bt.backtest(prices, pos).returns

    bumped = prices.copy()
    bumped.iloc[-1] *= 1.10
    rets2 = ind.log_returns(bumped)
    fvol2 = ind.ewma_volatility(rets2, lam=0.94)
    pos2 = bt.vol_target_position(bt.tsmom_signal(bumped, 60), fvol2, target_vol=0.10)
    after = bt.backtest(bumped, pos2).returns

    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1])
