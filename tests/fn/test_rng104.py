"""Tests for rng104.rangayyan_ch3_fourier_of_integral."""
import numpy as np
import pytest
from moirais.fn.rng104 import rangayyan_ch3_fourier_of_integral


def test_rng104_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_fourier_of_integral(X, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng104_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_fourier_of_integral(X, omega)
    assert isinstance(result, dict)
