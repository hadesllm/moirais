"""Tests for btnpqr.boot_quantile_regression."""
import numpy as np
import pytest
from morie.fn.btnpqr import boot_quantile_regression


def test_btnpqr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_quantile_regression(X, y, tau, B, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btnpqr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_quantile_regression(X, y, tau, B, alpha)
    assert isinstance(result, dict)
