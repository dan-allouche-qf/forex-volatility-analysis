import numpy as np
import pytest

from fxvol import stats_backtest as sb


def _normal_returns(mu, sigma, n, seed):
    return np.random.default_rng(seed).normal(mu, sigma, n)


def test_sharpe_bootstrap_ci_brackets_truth():
    # Daily mu/sigma chosen so the true annualized Sharpe ~ 1.0.
    sigma = 0.01
    mu = 1.0 * sigma / np.sqrt(252)
    r = _normal_returns(mu, sigma, 4000, seed=0)
    res = sb.sharpe_bootstrap(r, trading_days=252, n_boot=1000, seed=1)
    assert res["ci_low"] < res["sharpe"] < res["ci_high"]
    assert res["ci_low"] < 1.0 < res["ci_high"]


def test_sharpe_ci_narrows_with_sample_size():
    # More data -> a tighter Sharpe interval (estimation uncertainty shrinks ~1/sqrt(n)).
    sigma, mu = 0.01, 0.5 * 0.01 / np.sqrt(252)
    short = sb.sharpe_bootstrap(_normal_returns(mu, sigma, 400, 2), n_boot=800, seed=3)
    long = sb.sharpe_bootstrap(_normal_returns(mu, sigma, 6000, 2), n_boot=800, seed=3)
    assert (short["ci_high"] - short["ci_low"]) > (long["ci_high"] - long["ci_low"])


def test_psr_high_for_strong_sharpe():
    sigma = 0.01
    mu = 1.5 * sigma / np.sqrt(252)
    r = _normal_returns(mu, sigma, 3000, seed=4)
    assert sb.probabilistic_sharpe_ratio(r) > 0.99


def test_psr_low_for_negative_sharpe():
    sigma = 0.01
    mu = -1.5 * sigma / np.sqrt(252)
    r = _normal_returns(mu, sigma, 3000, seed=5)
    assert sb.probabilistic_sharpe_ratio(r) < 0.01


def test_psr_monotonic_in_mean():
    sigma = 0.01
    lo = sb.probabilistic_sharpe_ratio(_normal_returns(0.2 * sigma / np.sqrt(252), sigma, 2000, 7))
    hi = sb.probabilistic_sharpe_ratio(_normal_returns(0.9 * sigma / np.sqrt(252), sigma, 2000, 7))
    assert hi > lo


def test_deflated_more_conservative_than_psr():
    sigma = 0.01
    mu = 0.8 * sigma / np.sqrt(252)
    r = _normal_returns(mu, sigma, 2000, seed=6)
    psr = sb.probabilistic_sharpe_ratio(r)
    dsr = sb.deflated_sharpe_ratio(r, n_trials=20, var_trials_sr=(1.0 / 252))
    assert dsr < psr
    with pytest.raises(ValueError):
        sb.deflated_sharpe_ratio(r, n_trials=1, var_trials_sr=0.01)


def test_vol_model_scorecard_structure():
    from fxvol import data, models, risk
    from fxvol import indicators as ind
    from fxvol import preprocessing as pp
    r = ind.log_returns(pp.align_close(data.load_ohlc()))["EURUSD"]
    wf = models.walk_forward_variance(r, split=0.9, refit_every=30)
    sc = risk.vol_model_var_scorecard(r, wf, levels=(0.95, 0.99))
    assert set(sc["model"]) == {"garch", "ewma", "rolling", "naive"}
    assert (sc["breaches"] >= 0).all()
    assert len(sc) == 8
