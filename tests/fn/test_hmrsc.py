"""Tests for hmrsc.geron_randomized_search."""
import numpy as np
import pytest
from moirais.fn.hmrsc import geron_randomized_search


def test_hmrsc_basic():
    """Test basic functionality."""
    param_dist = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_randomized_search(param_dist, n_iter, X, y, estimator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrsc_edge():
    """Test edge cases."""
    param_dist = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_randomized_search(param_dist, n_iter, X, y, estimator)
    assert isinstance(result, dict)
