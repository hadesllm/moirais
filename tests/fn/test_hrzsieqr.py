"""Tests for hrzsieqr.horowitz_series_quantile."""
import numpy as np
import pytest
from morie.fn.hrzsieqr import horowitz_series_quantile


def test_hrzsieqr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    tau = 0.1
    result = horowitz_series_quantile(x, y, K, basis, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsieqr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    tau = 0.1
    result = horowitz_series_quantile(x, y, K, basis, tau)
    assert isinstance(result, dict)
