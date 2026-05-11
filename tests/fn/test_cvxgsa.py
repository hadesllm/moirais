"""Tests for cvxgsa.boyd_generalized_p."""
import numpy as np
import pytest
from morie.fn.cvxgsa import boyd_generalized_p


def test_cvxgsa_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boyd_generalized_p(x, y, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxgsa_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boyd_generalized_p(x, y, K)
    assert isinstance(result, dict)
