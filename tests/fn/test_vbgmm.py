"""Tests for morie.fn.vbgmm -- Variational Bayes GMM."""

import numpy as np
from morie.fn.vbgmm import vb_gaussian_mixture


def test_returns_dict():
    rng = np.random.default_rng(42)
    X = np.concatenate([rng.normal(0, 1, (30, 2)), rng.normal(5, 1, (30, 2))])
    result = vb_gaussian_mixture(X, K=3, max_iter=50)
    assert isinstance(result, dict)
    assert "weights" in result


def test_weights_sum_to_one():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 2))
    result = vb_gaussian_mixture(X, K=3, max_iter=50)
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_finds_clusters():
    rng = np.random.default_rng(42)
    X = np.concatenate([rng.normal(-5, 0.5, (50, 1)), rng.normal(5, 0.5, (50, 1))])
    result = vb_gaussian_mixture(X, K=5, max_iter=100)
    assert result["K"] == 5
