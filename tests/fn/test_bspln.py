"""Tests for morie.fn.bspln -- Bayesian spline."""

import numpy as np
from morie.fn.bspln import bayesian_spline


def test_returns_dict():
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + np.random.default_rng(42).standard_normal(50) * 0.2
    result = bayesian_spline(x, y, n_knots=5)
    assert isinstance(result, dict)
    assert "fitted" in result


def test_fitted_length():
    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    result = bayesian_spline(x, y, n_knots=5)
    assert len(result["fitted"]) == 30


def test_ci_contains_fitted():
    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    result = bayesian_spline(x, y, n_knots=5)
    for i in range(30):
        assert result["ci_lower"][i] <= result["ci_upper"][i]
