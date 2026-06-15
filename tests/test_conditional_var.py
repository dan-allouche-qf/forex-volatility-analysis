import numpy as np
import pytest
from scipy import stats

from fxvol import data, models, risk
from fxvol import indicators as ind
from fxvol import preprocessing as pp


def test_std_t_quantile_approaches_normal():
    # As nu -> inf the unit-variance t quantile converges to the normal quantile.
    assert risk._std_t_q(0.01, 1e6) == pytest.approx(stats.norm.ppf(0.01), abs=1e-2)


def test_std_t_es_deeper_than_var():
    a = 0.01
    assert risk._std_t_es(a, 6) > abs(risk._std_t_q(a, 6)) > 0


def test_pinball_minimized_at_true_quantile():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 100_000)
    a = 0.05
    q = np.quantile(r, a)
    assert risk.pinball_loss(r, q, a) < risk.pinball_loss(r, q - 0.5, a)
    assert risk.pinball_loss(r, q, a) < risk.pinball_loss(r, q + 0.5, a)


def test_acerbi_szekely_zero_under_correct_es():
    rng = np.random.default_rng(1)
    n = 200_000
    r = rng.normal(0, 0.01, n)
    level = 0.975
    a = 1 - level
    z = stats.norm.ppf(a)
    var = np.full(n, -(0.01 * z))
    es = np.full(n, 0.01 * stats.norm.pdf(z) / a)
    assert abs(risk.acerbi_szekely_z2(r, var, es, level)) < 0.1


def test_acerbi_szekely_negative_when_es_understated():
    rng = np.random.default_rng(2)
    n = 100_000
    r = rng.normal(0, 0.01, n)
    level = 0.975
    a = 1 - level
    z = stats.norm.ppf(a)
    var = np.full(n, -(0.01 * z))
    es = np.full(n, 0.5 * 0.01 * stats.norm.pdf(z) / a)  # ES halved -> understated risk
    assert risk.acerbi_szekely_z2(r, var, es, level) < -0.2


@pytest.fixture(scope="module")
def eur_garch_wf():
    r = ind.log_returns(pp.align_close(data.load_ohlc()))["EURUSD"]
    wf = models.walk_forward_garch(r, refit_every=20, dist="t",
                                   min_train=len(r) - 45, levels=(0.95, 0.99))
    return r, wf


def test_walk_forward_garch_columns(eur_garch_wf):
    _, wf = eur_garch_wf
    for c in ["sigma", "mu", "nu", "fhs_q_95", "fhs_es_95", "fhs_q_99", "fhs_es_99"]:
        assert c in wf.columns
    assert (wf["sigma"] > 0).all() and wf["nu"].gt(2).all()


def test_rolling_var_backtest_structure(eur_garch_wf):
    r, wf = eur_garch_wf
    sc = risk.rolling_var_backtest(r, wf, levels=(0.95, 0.99), window=250)
    assert set(sc["method"]) == {"hist", "t", "garch_t", "fhs"}
    assert len(sc) == 8
    assert (sc["breaches"] >= 0).all()
    assert (sc["pinball"] > 0).all()


def test_es_backtest_runs(eur_garch_wf):
    r, wf = eur_garch_wf
    out = risk.es_backtest_garch_t(r, wf, level=0.99, n_sim=200, seed=0)
    assert 0.0 <= out["p_value"] <= 1.0
    assert np.isfinite(out["Z2"])
