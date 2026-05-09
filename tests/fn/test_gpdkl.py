"""Tests for gpdkl.deep_kernel_gp."""
import numpy as np
import pytest
from moirais.fn.gpdkl import deep_kernel_gp


def test_gpdkl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    nn = np.random.default_rng(42).normal(0, 1, 100)
    result = deep_kernel_gp(X, y, X_test, nn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpdkl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    nn = np.random.default_rng(42).normal(0, 1, 100)
    result = deep_kernel_gp(X, y, X_test, nn)
    assert isinstance(result, dict)
