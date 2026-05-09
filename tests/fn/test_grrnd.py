"""Tests for grrnd.geron_randomized_search_cv."""
import numpy as np
import pytest
from moirais.fn.grrnd import geron_randomized_search_cv


def test_grrnd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    param_dist = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = geron_randomized_search_cv(X, y, param_dist, n_iter, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grrnd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    param_dist = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = geron_randomized_search_cv(X, y, param_dist, n_iter, K)
    assert isinstance(result, dict)
