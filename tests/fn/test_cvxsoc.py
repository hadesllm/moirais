"""Tests for cvxsoc.boyd_socp."""
import numpy as np
import pytest
from morie.fn.cvxsoc import boyd_socp


def test_cvxsoc_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = boyd_socp(f, A, b, c, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxsoc_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = boyd_socp(f, A, b, c, d)
    assert isinstance(result, dict)
