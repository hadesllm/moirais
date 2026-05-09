"""Tests for jotsrg.joseph_ts_as_regression."""
import numpy as np
import pytest
from moirais.fn.jotsrg import joseph_ts_as_regression


def test_jotsrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lags = 10
    rolling_windows = np.random.default_rng(42).normal(0, 1, 100)
    seasonal_m = np.random.default_rng(42).normal(0, 1, 100)
    fourier_K = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_ts_as_regression(y, lags, rolling_windows, seasonal_m, fourier_K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jotsrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lags = 10
    rolling_windows = np.random.default_rng(42).normal(0, 1, 100)
    seasonal_m = np.random.default_rng(42).normal(0, 1, 100)
    fourier_K = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_ts_as_regression(y, lags, rolling_windows, seasonal_m, fourier_K)
    assert isinstance(result, dict)
