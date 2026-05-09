"""Tests for rng232.rangayyan_ch4_homomorphic_log_fourier."""
import numpy as np
import pytest
from moirais.fn.rng232 import rangayyan_ch4_homomorphic_log_fourier


def test_rng232_basic():
    """Test basic functionality."""
    X_l = np.random.default_rng(42).normal(0, 1, 100)
    P_l = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_homomorphic_log_fourier(X_l, P_l, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng232_edge():
    """Test edge cases."""
    X_l = np.random.default_rng(42).normal(0, 1, 100)
    P_l = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_homomorphic_log_fourier(X_l, P_l, omega)
    assert isinstance(result, dict)
