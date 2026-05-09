"""Tests for quanrg.quantile_regression."""
import numpy as np
import pytest
from moirais.fn.quanrg import quantile_regression


def test_quanrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tau = 0.1
    result = quantile_regression(y, X, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_quanrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tau = 0.1
    result = quantile_regression(y, X, tau)
    assert isinstance(result, dict)
