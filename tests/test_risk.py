import numpy as np
import pandas as pd
import pytest

from fxvol import risk


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0, 0.01, 200_000))


def test_historical_var_matches_normal_quantile(normal_returns):
    v95 = risk.historical_var(normal_returns, 0.95)
    assert v95 == pytest.approx(1.645 * 0.01, rel=0.05)


def test_normal_var_es_ordering(normal_returns):
    v = risk.normal_var(normal_returns, 0.95)
    es = risk.normal_es(normal_returns, 0.95)
    assert es > v > 0  # ES is a deeper-tail average than VaR


def test_normal_and_historical_var_agree(normal_returns):
    assert risk.normal_var(normal_returns, 0.99) == pytest.approx(
        risk.historical_var(normal_returns, 0.99), rel=0.05
    )


def test_student_t_approaches_normal_for_normal_data(normal_returns):
    # Fitting a t to ~Gaussian data yields large df, so t-VaR ~ normal-VaR.
    assert risk.student_t_var(normal_returns, 0.99) == pytest.approx(
        risk.normal_var(normal_returns, 0.99), rel=0.1
    )
    assert risk.student_t_es(normal_returns, 0.99) > risk.student_t_var(normal_returns, 0.99)


def test_max_drawdown_known_path():
    # equity 1 -> 1.1 -> 0.55 -> 0.66 ; max DD = 0.55/1.1 - 1 = -0.5
    r = pd.Series([0.1, -0.5, 0.2])
    mdd = risk.max_drawdown(r)
    assert mdd["max_drawdown"] == pytest.approx(-0.5)
    assert mdd["trough"] == 1


def test_kupiec_passes_when_exceptions_match_expectation():
    n, level = 1000, 0.99
    x = int(n * (1 - level))  # exactly the expected number of breaches
    out = risk.kupiec_pof(n, x, level)
    assert out["p_value"] > 0.9  # should not reject correct coverage


def test_kupiec_rejects_far_too_many_exceptions():
    out = risk.kupiec_pof(1000, 100, 0.99)  # 10x expected
    assert out["p_value"] < 0.01


def test_christoffersen_independent_series_not_rejected():
    rng = np.random.default_rng(0)
    h = pd.Series((rng.random(5000) < 0.05).astype(int))  # iid breaches
    out = risk.christoffersen(h, 0.95)
    assert out["p_ind"] > 0.05


def test_christoffersen_detects_clustering():
    # All breaches bunched together -> independence strongly rejected.
    h = pd.Series([0] * 200 + [1] * 20 + [0] * 200)
    out = risk.christoffersen(h, 0.95)
    assert out["p_ind"] < 0.05


def test_sharpe_zero_for_zero_mean():
    r = pd.Series([0.01, -0.01, 0.01, -0.01] * 50)
    assert risk.sharpe(r) == pytest.approx(0.0, abs=1e-9)


def test_risk_table_shape_and_signs():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"A": rng.normal(0, 0.01, 1000), "B": rng.normal(0, 0.02, 1000)})
    tbl = risk.risk_table(df, levels=(0.95, 0.99))
    assert set(tbl.index) == {"A", "B"}
    assert (tbl["ann_vol"] > 0).all()
    assert (tbl["ES_t_99"] >= tbl["VaR_t_99"]).all()
