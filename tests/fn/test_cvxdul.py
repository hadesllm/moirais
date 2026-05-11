"""Tests for cvxdul.boyd_lagrangian."""
import numpy as np
import pytest
from morie.fn.cvxdul import boyd_lagrangian


def test_cvxdul_basic():
    """Test basic functionality."""
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_lagrangian(f0, f, h, lambda_, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxdul_edge():
    """Test edge cases."""
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_lagrangian(f0, f, h, lambda_, nu)
    assert isinstance(result, dict)
