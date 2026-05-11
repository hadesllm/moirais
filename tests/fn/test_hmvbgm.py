"""Tests for hmvbgm.geron_variational_bayes_gmm."""
import numpy as np
import pytest
from morie.fn.hmvbgm import geron_variational_bayes_gmm


def test_hmvbgm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_variational_bayes_gmm(X, n_components, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvbgm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_variational_bayes_gmm(X, n_components, max_iter)
    assert isinstance(result, dict)
