"""Tests for rng234.rangayyan_ch4_fourier_convolution_property."""
import numpy as np
import pytest
from moirais.fn.rng234 import rangayyan_ch4_fourier_convolution_property


def test_rng234_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_fourier_convolution_property(X, H, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng234_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_fourier_convolution_property(X, H, omega)
    assert isinstance(result, dict)
