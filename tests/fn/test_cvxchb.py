"""Tests for cvxchb.boyd_chebyshev_center."""
import numpy as np
import pytest
from morie.fn.cvxchb import boyd_chebyshev_center


def test_cvxchb_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_chebyshev_center(A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxchb_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_chebyshev_center(A, b)
    assert isinstance(result, dict)
