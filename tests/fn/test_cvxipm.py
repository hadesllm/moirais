"""Tests for cvxipm.boyd_interior_point."""
import numpy as np
import pytest
from moirais.fn.cvxipm import boyd_interior_point


def test_cvxipm_basic():
    """Test basic functionality."""
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_interior_point(f0, f, x0, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxipm_edge():
    """Test edge cases."""
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_interior_point(f0, f, x0, t)
    assert isinstance(result, dict)
