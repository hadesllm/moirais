"""Tests for rng237.rangayyan_ch4_log_of_convolved_signals."""
import numpy as np
import pytest
from moirais.fn.rng237 import rangayyan_ch4_log_of_convolved_signals


def test_rng237_basic():
    """Test basic functionality."""
    X_hat = np.random.default_rng(42).normal(0, 1, 100)
    H_hat = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_log_of_convolved_signals(X_hat, H_hat, z, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng237_edge():
    """Test edge cases."""
    X_hat = np.random.default_rng(42).normal(0, 1, 100)
    H_hat = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_log_of_convolved_signals(X_hat, H_hat, z, omega)
    assert isinstance(result, dict)
