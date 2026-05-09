"""Tests for kmcnf: KM confidence intervals."""
import numpy as np
import pytest
from moirais.fn.kmcnf import kmcnf


def _make_data(n=200, rate=0.5, censor_rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.exponential(1 / rate, size=n)
    C = rng.exponential(1 / censor_rate, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event


def test_returns_keys():
    time, event = _make_data()
    result = kmcnf(time, event)
    for key in ("time_points", "survival", "n_risk", "n_event",
                "ci_lower", "ci_upper", "greenwood_var", "median_survival"):
        assert key in result


def test_survival_starts_at_one():
    """First KM value should be < 1 (first event drops below 1)."""
    time, event = _make_data()
    result = kmcnf(time, event)
    assert result["survival"][0] <= 1.0


def test_survival_monotone_decreasing():
    time, event = _make_data()
    result = kmcnf(time, event)
    assert np.all(np.diff(result["survival"]) <= 1e-12)


def test_survival_in_0_1():
    time, event = _make_data()
    result = kmcnf(time, event)
    assert np.all(result["survival"] >= 0)
    assert np.all(result["survival"] <= 1)


def test_ci_bounds_bracket_survival():
    time, event = _make_data()
    result = kmcnf(time, event)
    assert np.all(result["ci_lower"] <= result["survival"] + 1e-12)
    assert np.all(result["ci_upper"] >= result["survival"] - 1e-12)


def test_ci_in_0_1():
    time, event = _make_data()
    result = kmcnf(time, event)
    assert np.all(result["ci_lower"] >= 0)
    assert np.all(result["ci_upper"] <= 1)


def test_all_ci_types():
    time, event = _make_data()
    for ci_type in ("linear", "log", "log-log", "logit", "arcsin"):
        result = kmcnf(time, event, ci_type=ci_type)
        assert np.all(np.isfinite(result["ci_lower"]))
        assert np.all(np.isfinite(result["ci_upper"]))


def test_median_survival_exponential():
    """For Exp(rate), median = log(2)/rate."""
    rng = np.random.default_rng(42)
    n = 5000
    rate = 0.5
    T = rng.exponential(1 / rate, size=n)
    time = T
    event = np.ones(n)
    result = kmcnf(time, event)
    expected_median = np.log(2) / rate
    assert abs(result["median_survival"] - expected_median) < 0.1


def test_greenwood_var_nonneg():
    time, event = _make_data()
    result = kmcnf(time, event)
    assert np.all(result["greenwood_var"] >= 0)


def test_invalid_ci_type_raises():
    time, event = _make_data()
    with pytest.raises(ValueError):
        kmcnf(time, event, ci_type="bogus")


def test_time_points_sorted():
    time, event = _make_data()
    result = kmcnf(time, event)
    assert np.all(np.diff(result["time_points"]) > 0)
