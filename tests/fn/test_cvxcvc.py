"""Tests for cvxcvc.boyd_convex_combination."""
import numpy as np
import pytest
from morie.fn.cvxcvc import boyd_convex_combination


def test_cvxcvc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = boyd_convex_combination(x, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxcvc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = boyd_convex_combination(x, theta)
    assert isinstance(result, dict)
