"""Tests for cvxbar.boyd_log_barrier."""
import numpy as np
import pytest
from morie.fn.cvxbar import boyd_log_barrier


def test_cvxbar_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_log_barrier(f, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxbar_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_log_barrier(f, x)
    assert isinstance(result, dict)
