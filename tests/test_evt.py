import numpy as np
import pytest
from scipy import stats

from fxvol import evt


def test_gpd_recovers_known_shape():
    # Exceedances of a heavy-tailed sample over a high threshold are ~GPD; the
    # fitted shape should be positive for a Student-t tail.
    rng = np.random.default_rng(0)
    losses = -stats.t.rvs(4, size=200_000, random_state=rng)  # right tail = losses
    losses = losses[losses > 0]
    fit = evt.gpd_fit(losses, threshold_quantile=0.95)
    assert fit["shape"] > 0.0          # heavy tail
    assert fit["scale"] > 0.0
    assert fit["n_exceed"] > 100


def test_pot_var_es_ordering_and_tail_growth():
    rng = np.random.default_rng(1)
    losses = -stats.t.rvs(5, size=200_000, random_state=rng)
    losses = losses[losses > 0]
    v99 = evt.pot_var_es(losses, level=0.99, threshold_quantile=0.90)
    v995 = evt.pot_var_es(losses, level=0.995, threshold_quantile=0.90)
    assert v99["ES"] > v99["VaR"] > 0
    assert v995["VaR"] > v99["VaR"]     # deeper level -> larger VaR


def test_conditional_pot_scales_with_sigma():
    rng = np.random.default_rng(2)
    losses = -stats.t.rvs(6, size=100_000, random_state=rng)
    losses = losses[losses > 0]
    a = evt.conditional_pot_var_es(0.01, losses, level=0.99)
    b = evt.conditional_pot_var_es(0.02, losses, level=0.99)
    assert b["cond_VaR"] == pytest.approx(2.0 * a["cond_VaR"], rel=1e-9)


def test_mean_excess_monotone_grid():
    rng = np.random.default_rng(3)
    losses = -stats.t.rvs(5, size=50_000, random_state=rng)
    losses = losses[losses > 0]
    me = evt.mean_excess(losses, n_points=20)
    assert len(me) > 5 and (me["n_exceed"].diff().dropna() <= 0).all()
