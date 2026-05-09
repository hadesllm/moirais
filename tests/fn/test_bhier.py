"""Tests for moirais.fn.bhier -- Bayesian hierarchical model."""

import numpy as np
from moirais.fn.bhier import bayesian_hierarchical


def test_returns_dict():
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 20), rng.normal(1, 1, 20), rng.normal(2, 1, 20)]
    result = bayesian_hierarchical(groups, n_iter=500)
    assert isinstance(result, dict)
    assert "mu_samples" in result


def test_mu_mean_reasonable():
    rng = np.random.default_rng(42)
    groups = [rng.normal(5, 1, 30) for _ in range(5)]
    result = bayesian_hierarchical(groups, n_iter=1000)
    assert abs(result["mu_mean"] - 5.0) < 2.0


def test_j_count():
    groups = [[1, 2, 3], [4, 5, 6]]
    result = bayesian_hierarchical(groups, n_iter=200)
    assert result["J"] == 2
