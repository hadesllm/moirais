"""Tests for rng125.rangayyan_ch3_butterworth_pole_positions."""
import numpy as np
import pytest
from morie.fn.rng125 import rangayyan_ch3_butterworth_pole_positions


def test_rng125_basic():
    """Test basic functionality."""
    Omega_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    k = 5
    result = rangayyan_ch3_butterworth_pole_positions(Omega_c, N, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng125_edge():
    """Test edge cases."""
    Omega_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    k = 5
    result = rangayyan_ch3_butterworth_pole_positions(Omega_c, N, k)
    assert isinstance(result, dict)
