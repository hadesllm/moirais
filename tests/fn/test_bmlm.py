"""Tests for morie.fn.bmlm -- Bayesian multilevel model."""

import numpy as np
from morie.fn.bmlm import bayesian_multilevel


def test_returns_dict():
    result = bayesian_multilevel([1, 2, 3], [0.5, 0.5, 0.5], n_iter=500)
    assert isinstance(result, dict)
    assert "grand_mean" in result
    assert "theta_means" in result


def test_shrinkage_towards_grand_mean():
    means = [0.0, 10.0]
    ses = [1.0, 1.0]
    result = bayesian_multilevel(means, ses, n_iter=5000, seed=42)
    assert result["theta_means"][0] > 0.0
    assert result["theta_means"][1] < 10.0


def test_grand_mean_near_average():
    means = [5.0, 5.0, 5.0]
    ses = [0.5, 0.5, 0.5]
    result = bayesian_multilevel(means, ses, n_iter=5000, seed=42)
    assert abs(result["grand_mean"] - 5.0) < 1.0


def test_shrinkage_factors():
    result = bayesian_multilevel([1, 2, 3], [0.5, 0.5, 0.5], n_iter=2000)
    assert np.all(result["shrinkage"] >= 0)
    assert np.all(result["shrinkage"] <= 1)


def test_mismatched_lengths():
    try:
        bayesian_multilevel([1, 2], [0.5])
        assert False
    except ValueError:
        pass


def test_invalid_se():
    try:
        bayesian_multilevel([1, 2], [0.5, -1])
        assert False
    except ValueError:
        pass
