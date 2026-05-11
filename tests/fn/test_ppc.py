"""Tests for morie.fn.ppc -- posterior predictive check."""

import numpy as np
from morie.fn.ppc import posterior_predictive_check


def test_returns_dict():
    result = posterior_predictive_check([1, 2, 3], np.array([[1, 2, 3], [2, 3, 4]]))
    assert isinstance(result, dict)
    assert "bayesian_p" in result


def test_perfect_fit():
    obs = np.array([0.0, 0.0, 0.0])
    reps = np.zeros((100, 3))
    result = posterior_predictive_check(obs, reps)
    assert result["bayesian_p"] == 1.0


def test_extreme_misfit():
    obs = np.array([100.0, 100.0])
    reps = np.zeros((100, 2))
    result = posterior_predictive_check(obs, reps)
    assert result["bayesian_p"] == 0.0


def test_various_statistics():
    rng = np.random.default_rng(42)
    obs = rng.normal(0, 1, 50)
    reps = rng.normal(0, 1, (200, 50))
    for stat in ["mean", "var", "max", "min", "median"]:
        result = posterior_predictive_check(obs, reps, test_statistic=stat)
        assert 0 <= result["bayesian_p"] <= 1


def test_1d_replications():
    obs = np.array([2.0, 3.0, 4.0])
    T_reps = np.array([2.5, 3.5, 2.8, 3.1, 2.9])
    result = posterior_predictive_check(obs, T_reps)
    expected_p = np.mean(T_reps >= np.mean(obs))
    np.testing.assert_allclose(result["bayesian_p"], expected_p)


def test_unknown_statistic():
    try:
        posterior_predictive_check([1], [[1]], test_statistic="invalid")
        assert False
    except ValueError:
        pass


def test_quantiles():
    rng = np.random.default_rng(42)
    obs = rng.normal(0, 1, 30)
    reps = rng.normal(0, 1, (500, 30))
    result = posterior_predictive_check(obs, reps)
    q = result["T_rep_quantiles"]
    assert q["q025"] <= q["q500"] <= q["q975"]
