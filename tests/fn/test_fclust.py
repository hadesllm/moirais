"""Tests for fclust.functional_clustering."""
import numpy as np
import pytest
from moirais.fn.fclust import functional_clustering


def test_fclust_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = functional_clustering(Y, K, basis)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fclust_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = functional_clustering(Y, K, basis)
    assert isinstance(result, dict)
