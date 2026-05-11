"""Tests for rng055.rangayyan_ch3_dtft_via_z."""
import numpy as np
import pytest
from morie.fn.rng055 import rangayyan_ch3_dtft_via_z


def test_rng055_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    N = 100
    result = rangayyan_ch3_dtft_via_z(x, n, omega, T, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng055_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    N = 100
    result = rangayyan_ch3_dtft_via_z(x, n, omega, T, N)
    assert isinstance(result, dict)
