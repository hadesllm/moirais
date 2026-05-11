"""Tests for morie.fn.emvb -- EM variational Bayes."""

import numpy as np
from morie.fn.emvb import em_variational_bayes


def test_returns_dict():
    rng = np.random.default_rng(42)
    data = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
    result = em_variational_bayes(data, n_components=2, max_iter=50)
    assert isinstance(result, dict)
    assert "means" in result
    assert "weights" in result


def test_weights_sum_to_one():
    rng = np.random.default_rng(42)
    data = np.concatenate([rng.normal(-3, 0.5, 40), rng.normal(3, 0.5, 40)])
    result = em_variational_bayes(data, n_components=2)
    np.testing.assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-10)


def test_separates_clusters():
    rng = np.random.default_rng(123)
    data = np.concatenate([rng.normal(-5, 0.5, 200), rng.normal(5, 0.5, 200)])
    result = em_variational_bayes(data, n_components=2, max_iter=500, seed=123)
    means_sorted = np.sort(result["means"])
    assert means_sorted[1] - means_sorted[0] > 3.0


def test_elbo_nondecreasing():
    rng = np.random.default_rng(42)
    data = np.concatenate([rng.normal(0, 1, 50), rng.normal(4, 1, 50)])
    result = em_variational_bayes(data, n_components=2, max_iter=100)
    h = result["elbo_history"]
    for i in range(1, len(h)):
        assert h[i] >= h[i - 1] - 1e-6


def test_too_few_points():
    try:
        em_variational_bayes([1.0], n_components=2)
        assert False
    except ValueError:
        pass
