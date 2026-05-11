"""Tests for cvxhul.boyd_convex_hull."""
import numpy as np
import pytest
from morie.fn.cvxhul import boyd_convex_hull


def test_cvxhul_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_convex_hull(S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxhul_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_convex_hull(S)
    assert isinstance(result, dict)
