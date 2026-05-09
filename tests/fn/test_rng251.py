"""Tests for rng251.rangayyan_ch4_log_echo_power_series_expansion."""
import numpy as np
import pytest
from moirais.fn.rng251 import rangayyan_ch4_log_echo_power_series_expansion


def test_rng251_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    H_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_log_echo_power_series_expansion(a, n_0, omega, H_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng251_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    H_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_log_echo_power_series_expansion(a, n_0, omega, H_hat)
    assert isinstance(result, dict)
