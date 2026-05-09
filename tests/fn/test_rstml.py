"""Tests for rstml: restricted mean survival time."""
import numpy as np
import pytest
from moirais.fn.rstml import rstml


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.exponential(2.0, size=n)
    C = rng.exponential(5.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event


def test_returns_keys():
    time, event = _make_data()
    result = rstml(time, event)
    for key in ("rmst", "se", "ci_lower", "ci_upper", "tau"):
        assert key in result


def test_rmst_positive():
    time, event = _make_data()
    result = rstml(time, event)
    assert result["rmst"] > 0


def test_rmst_leq_tau():
    """RMST <= tau always."""
    time, event = _make_data()
    result = rstml(time, event)
    assert result["rmst"] <= result["tau"] + 1e-10


def test_rmst_exponential_analytic():
    """For exp(rate=1) with no censoring, RMST(tau) = 1 - exp(-tau)."""
    rng = np.random.default_rng(42)
    n = 10000
    T = rng.exponential(1.0, size=n)
    time = T
    event = np.ones(n)
    tau = 2.0
    result = rstml(time, event, tau=tau)
    expected = 1 - np.exp(-tau)  # integral_0^tau exp(-t) dt
    assert abs(result["rmst"] - expected) < 0.05


def test_ci_bounds_ordered():
    time, event = _make_data()
    result = rstml(time, event)
    assert result["ci_lower"] <= result["rmst"] <= result["ci_upper"]


def test_two_group_rmst_diff():
    rng = np.random.default_rng(1)
    n = 200
    T0 = rng.exponential(1.0, size=n)
    T1 = rng.exponential(2.0, size=n)
    C = rng.exponential(5.0, size=2 * n)
    time = np.concatenate([np.minimum(T0, C[:n]), np.minimum(T1, C[n:])])
    event = np.concatenate([(T0 <= C[:n]).astype(float), (T1 <= C[n:]).astype(float)])
    group = np.array([0] * n + [1] * n)
    result = rstml(time, event, group=group)
    assert "rmst_diff" in result
    # Group 1 has longer survival => positive RMST difference
    assert result["rmst_diff"] > 0


def test_two_group_pvalue():
    rng = np.random.default_rng(3)
    n = 300
    T0 = rng.exponential(0.5, size=n)
    T1 = rng.exponential(2.0, size=n)
    C = rng.exponential(5.0, size=2 * n)
    time = np.concatenate([np.minimum(T0, C[:n]), np.minimum(T1, C[n:])])
    event = np.concatenate([(T0 <= C[:n]).astype(float), (T1 <= C[n:]).astype(float)])
    group = np.array([0] * n + [1] * n)
    result = rstml(time, event, group=group)
    assert result["p_value"] < 0.05


def test_invalid_time_raises():
    with pytest.raises(ValueError):
        rstml(np.array([-1.0, 2.0, 3.0]), np.array([1, 0, 1]))
