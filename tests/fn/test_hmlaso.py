"""Tests for hmlaso.geron_lasso_cost."""
import numpy as np
import pytest
from morie.fn.hmlaso import geron_lasso_cost


def test_hmlaso_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    result = geron_lasso_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlaso_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    result = geron_lasso_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
