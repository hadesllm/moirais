"""Tests for cvxlsq.boyd_least_squares."""
import numpy as np
import pytest
from morie.fn.cvxlsq import boyd_least_squares


def test_cvxlsq_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_least_squares(A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxlsq_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_least_squares(A, b)
    assert isinstance(result, dict)
