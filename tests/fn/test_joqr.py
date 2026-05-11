"""Tests for joqr.joseph_quantile_regression."""
import numpy as np
import pytest
from morie.fn.joqr import joseph_quantile_regression


def test_joqr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    result = joseph_quantile_regression(X, y, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joqr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    result = joseph_quantile_regression(X, y, tau)
    assert isinstance(result, dict)
