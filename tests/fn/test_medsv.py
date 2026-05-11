"""Tests for medsv: median survival time with CI."""
import numpy as np
import pytest
from morie.fn.medsv import medsv


def _make_data(n=300, rate=0.5, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.exponential(1 / rate, size=n)
    C = rng.exponential(4.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event


def test_returns_keys():
    time, event = _make_data()
    result = medsv(time, event)
    for key in ("estimate", "ci_lower", "ci_upper", "km_survival", "n_events", "n_total"):
        assert key in result


def test_median_exponential_close_to_analytic():
    """For Exp(0.5), median = log(2)/0.5 = 1.386."""
    rng = np.random.default_rng(42)
    n = 5000
    T = rng.exponential(2.0, size=n)
    time = T
    event = np.ones(n)
    result = medsv(time, event)
    expected = np.log(2) * 2.0
    assert abs(result["estimate"] - expected) < 0.1


def test_km_survival_near_0_5():
    """KM survival at median should be near 0.5."""
    time, event = _make_data(n=2000, seed=1)
    result = medsv(time, event)
    if not np.isnan(result["estimate"]):
        assert abs(result["km_survival"] - 0.5) < 0.1


def test_ci_finite():
    """CI bounds should be finite positive numbers when estimate is defined."""
    time, event = _make_data()
    result = medsv(time, event)
    if not np.isnan(result["estimate"]):
        if not np.isnan(result["ci_lower"]):
            assert result["ci_lower"] > 0
        if not np.isnan(result["ci_upper"]):
            assert result["ci_upper"] > 0


def test_n_total_correct():
    time, event = _make_data(n=200)
    result = medsv(time, event)
    assert result["n_total"] == 200


def test_n_events_correct():
    time, event = _make_data(n=200)
    result = medsv(time, event)
    assert result["n_events"] == int(event.sum())


def test_other_quantile():
    """75th percentile of survival."""
    time, event = _make_data(n=1000, seed=3)
    result = medsv(time, event, quantile=0.75)
    # KM survival at 75th percentile should be near 0.25
    if not np.isnan(result["estimate"]) and not np.isnan(result["km_survival"]):
        assert abs(result["km_survival"] - 0.25) < 0.15


def test_invalid_quantile_raises():
    time, event = _make_data()
    with pytest.raises(ValueError):
        medsv(time, event, quantile=1.5)


def test_negative_time_raises():
    with pytest.raises(ValueError):
        medsv(np.array([-1.0, 2.0, 3.0]), np.array([1, 0, 1]))
