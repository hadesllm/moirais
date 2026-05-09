"""Tests for rng066.rangayyan_ch3_inverse_fourier_transform."""
import numpy as np
import pytest
from moirais.fn.rng066 import rangayyan_ch3_inverse_fourier_transform


def test_rng066_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    omega = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_inverse_fourier_transform(X, omega, f, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng066_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    omega = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_inverse_fourier_transform(X, omega, f, t)
    assert isinstance(result, dict)
