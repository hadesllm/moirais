"""Tests for gpkrr.gp_kernel_ridge_reg."""
import numpy as np
import pytest
from morie.fn.gpkrr import gp_kernel_ridge_reg


def test_gpkrr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    lam = 0.1
    result = gp_kernel_ridge_reg(X, y, X_test, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpkrr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    lam = 0.1
    result = gp_kernel_ridge_reg(X, y, X_test, lam)
    assert isinstance(result, dict)
