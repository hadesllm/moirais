"""Tests for moirais.fn.empby -- Empirical Bayes."""

import numpy as np
from moirais.fn.empby import empirical_bayes


def test_returns_dict():
    estimates = np.random.default_rng(42).standard_normal(20)
    ses = np.abs(np.random.default_rng(43).standard_normal(20)) + 0.1
    result = empirical_bayes(estimates, ses)
    assert isinstance(result, dict)
    assert "shrunk_estimates" in result


def test_shrinkage_toward_zero():
    estimates = np.array([5.0, -5.0, 0.1, -0.1])
    ses = np.array([1.0, 1.0, 1.0, 1.0])
    result = empirical_bayes(estimates, ses)
    for i in range(4):
        assert abs(result["shrunk_estimates"][i]) <= abs(estimates[i]) + 0.01


def test_tau2_positive():
    estimates = np.random.default_rng(42).standard_normal(30) * 3
    ses = np.ones(30)
    result = empirical_bayes(estimates, ses)
    assert result["tau2"] > 0
