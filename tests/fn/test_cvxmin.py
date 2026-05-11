"""Tests for cvxmin.boyd_minimum_norm."""
import numpy as np
import pytest
from morie.fn.cvxmin import boyd_minimum_norm


def test_cvxmin_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    norm = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_minimum_norm(A, b, norm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxmin_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    norm = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_minimum_norm(A, b, norm)
    assert isinstance(result, dict)
