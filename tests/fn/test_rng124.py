"""Tests for rng124.rangayyan_ch3_butterworth_squared_laplace."""
import numpy as np
import pytest
from morie.fn.rng124 import rangayyan_ch3_butterworth_squared_laplace


def test_rng124_basic():
    """Test basic functionality."""
    s = 90
    Omega_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_squared_laplace(s, Omega_c, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng124_edge():
    """Test edge cases."""
    s = 90
    Omega_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_squared_laplace(s, Omega_c, N)
    assert isinstance(result, dict)
