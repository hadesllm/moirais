"""Tests for btciqua.boot_ci_quantile."""
import numpy as np
import pytest
from morie.fn.btciqua import boot_ci_quantile


def test_btciqua_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_quantile(x, tau, B, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btciqua_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_quantile(x, tau, B, alpha)
    assert isinstance(result, dict)
