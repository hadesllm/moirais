"""Tests for elasrg.elastic_net_regression."""
import numpy as np
import pytest
from morie.fn.elasrg import elastic_net_regression


def test_elasrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lambda1 = np.random.default_rng(42).normal(0, 1, 100)
    lambda2 = np.random.default_rng(42).normal(0, 1, 100)
    result = elastic_net_regression(y, X, lambda1, lambda2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_elasrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lambda1 = np.random.default_rng(42).normal(0, 1, 100)
    lambda2 = np.random.default_rng(42).normal(0, 1, 100)
    result = elastic_net_regression(y, X, lambda1, lambda2)
    assert isinstance(result, dict)
