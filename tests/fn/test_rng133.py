"""Tests for rng133.rangayyan_ch3_butterworth_lowpass_direct_specification."""
import numpy as np
import pytest
from moirais.fn.rng133 import rangayyan_ch3_butterworth_lowpass_direct_specification


def test_rng133_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    omega_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_lowpass_direct_specification(omega, omega_c, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng133_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    omega_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_lowpass_direct_specification(omega, omega_c, N)
    assert isinstance(result, dict)
