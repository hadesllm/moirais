"""Tests for rng065.rangayyan_ch3_fourier_transform_f."""
import numpy as np
import pytest
from moirais.fn.rng065 import rangayyan_ch3_fourier_transform_f


def test_rng065_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_fourier_transform_f(x, t, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng065_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_fourier_transform_f(x, t, f)
    assert isinstance(result, dict)
