import numpy as np
import pytest

from fxvol import data, models
from fxvol import indicators as ind


def test_qlike_and_mse_zero_when_perfect():
    f = np.array([0.5, 1.0, 2.0])
    assert models.qlike(f, f) == pytest.approx(np.zeros(3))
    assert models.mse(f, f) == pytest.approx(np.zeros(3))


def test_diebold_mariano_sign():
    # model A strictly better (lower loss) than B -> negative DM stat
    loss_a = np.full(500, 1.0)
    loss_b = np.full(500, 2.0)
    out = models.diebold_mariano(loss_a, loss_b)
    assert out["DM_stat"] < 0


def test_garch_fit_persistence_in_unit_interval():
    long = data.load_ohlc()
    close = long.pivot(index="Date", columns="pair", values="Close")["EURUSD"].sort_index()
    r = ind.log_returns(close)
    fit = models.fit_garch(r, dist="t")
    assert 0.0 < fit.persistence < 1.0
    assert fit.long_run_vol > 0
    assert fit.cond_vol.notna().all()


def test_walk_forward_scores_finite():
    long = data.load_ohlc()
    close = long.pivot(index="Date", columns="pair", values="Close")["EURUSD"].sort_index()
    r = ind.log_returns(close)
    wf = models.walk_forward_variance(r, split=0.92, refit_every=50)  # few steps -> fast
    assert {"garch", "ewma", "rolling", "naive", "realized"}.issubset(wf.columns)
    scores = models.score_forecasts(wf)
    assert np.isfinite(scores.values).all()
