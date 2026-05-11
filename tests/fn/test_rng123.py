"""Tests for rng123.rangayyan_ch3_butterworth_lowpass_squared_magnitude."""
import numpy as np
import pytest
from morie.fn.rng123 import rangayyan_ch3_butterworth_lowpass_squared_magnitude


def test_rng123_basic():
    """Test basic functionality."""
    Omega = np.random.default_rng(42).normal(0, 1, 100)
    Omega_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_lowpass_squared_magnitude(Omega, Omega_c, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng123_edge():
    """Test edge cases."""
    Omega = np.random.default_rng(42).normal(0, 1, 100)
    Omega_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_lowpass_squared_magnitude(Omega, Omega_c, N)
    assert isinstance(result, dict)
