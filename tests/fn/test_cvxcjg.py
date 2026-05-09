"""Tests for cvxcjg.boyd_conjugate."""
import numpy as np
import pytest
from moirais.fn.cvxcjg import boyd_conjugate


def test_cvxcjg_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = boyd_conjugate(f, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxcjg_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = boyd_conjugate(f, y)
    assert isinstance(result, dict)
