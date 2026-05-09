"""Tests for rng064.rangayyan_ch3_fourier_transform_omega."""
import numpy as np
import pytest
from moirais.fn.rng064 import rangayyan_ch3_fourier_transform_omega


def test_rng064_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_fourier_transform_omega(x, t, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng064_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_fourier_transform_omega(x, t, omega)
    assert isinstance(result, dict)
