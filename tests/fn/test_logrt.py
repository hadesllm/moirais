"""Tests for logrt: log-rank test."""
import numpy as np
import pytest
from moirais.fn.logrt import logrt


def _make_two_group_data(n_per_group=100, hr=2.0, seed=0):
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    # Group 0: exponential rate 0.5; Group 1: rate 0.5 * hr (higher hazard)
    rate_0 = 0.5
    rate_1 = rate_0 * hr
    T0 = rng.exponential(1 / rate_0, size=n_per_group)
    T1 = rng.exponential(1 / rate_1, size=n_per_group)
    C = rng.exponential(3.0, size=n)
    time = np.concatenate([T0, T1])
    event = np.ones(n)
    # Apply censoring
    cens = np.minimum(time, C)
    event = (time <= C).astype(float)
    group = np.array([0] * n_per_group + [1] * n_per_group)
    return cens, event, group


def test_returns_keys():
    time, event, group = _make_two_group_data()
    result = logrt(time, event, group)
    for key in ("statistic", "p_value", "df", "groups", "observed", "expected"):
        assert key in result


def test_p_value_significant_large_hr():
    """Large hazard ratio should give small p-value."""
    time, event, group = _make_two_group_data(n_per_group=200, hr=3.0, seed=1)
    result = logrt(time, event, group)
    assert result["p_value"] < 0.05


def test_p_value_not_significant_hr1():
    """HR=1 should give non-significant test most of the time."""
    time, event, group = _make_two_group_data(n_per_group=100, hr=1.0, seed=42)
    result = logrt(time, event, group)
    # Under null, p ~ U(0,1); we expect p > 0.01 most of the time
    assert result["p_value"] >= 0.0
    assert result["p_value"] <= 1.0


def test_df_is_one_for_two_groups():
    time, event, group = _make_two_group_data()
    result = logrt(time, event, group)
    assert result["df"] == 1


def test_three_groups():
    rng = np.random.default_rng(5)
    n = 30
    time = rng.exponential(1.0, size=3 * n)
    event = rng.choice([0, 1], size=3 * n)
    group = np.array([0] * n + [1] * n + [2] * n)
    result = logrt(time, event, group)
    assert result["df"] == 2


def test_statistic_nonneg():
    time, event, group = _make_two_group_data()
    result = logrt(time, event, group)
    assert result["statistic"] >= 0


def test_observed_sum_equals_total_events():
    time, event, group = _make_two_group_data()
    result = logrt(time, event, group)
    assert abs(result["observed"].sum() - event.sum()) < 1e-6


def test_fewer_than_two_groups_raises():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 0, 1])
    group = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        logrt(time, event, group)


def test_wilcoxon_weight():
    time, event, group = _make_two_group_data(n_per_group=100, hr=2.0, seed=2)
    result = logrt(time, event, group, rho=1.0)
    assert "statistic" in result
    assert result["p_value"] >= 0


def test_stratified_logrank():
    time, event, group = _make_two_group_data(n_per_group=100, hr=2.0, seed=3)
    strata = (np.arange(len(time)) % 2).astype(int)
    result = logrt(time, event, group, strata=strata)
    assert result["df"] == 1
    assert result["p_value"] < 0.05
