"""Tests for moirais.fn.bbeta -- Bayesian beta-binomial model."""

import numpy as np
from moirais.fn.bbeta import bayesian_beta_binomial


def test_returns_dict():
    result = bayesian_beta_binomial([5, 10], [10, 20], n_grid=50)
    assert isinstance(result, dict)
    assert "group_means" in result
    assert "hyper_a" in result


def test_group_means_in_01():
    result = bayesian_beta_binomial([3, 7, 5], [10, 10, 10], n_grid=50)
    assert np.all(result["group_means"] >= 0)
    assert np.all(result["group_means"] <= 1)


def test_ci_contains_mean():
    result = bayesian_beta_binomial([5, 10, 15], [20, 20, 20], n_grid=50)
    for i in range(3):
        assert result["group_ci_lower"][i] <= result["group_means"][i]
        assert result["group_means"][i] <= result["group_ci_upper"][i]


def test_shrinkage():
    k = [0, 10]
    n = [10, 10]
    result = bayesian_beta_binomial(k, n, n_grid=50)
    assert result["group_means"][0] > 0.0
    assert result["group_means"][1] < 1.0


def test_mismatched_lengths():
    try:
        bayesian_beta_binomial([1, 2], [10])
        assert False
    except ValueError:
        pass
