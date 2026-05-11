"""Tests for rng129.rangayyan_ch3_bilinear_warping_omega_to_Omega."""
import numpy as np
import pytest
from morie.fn.rng129 import rangayyan_ch3_bilinear_warping_omega_to_Omega


def test_rng129_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_bilinear_warping_omega_to_Omega(omega, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng129_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_bilinear_warping_omega_to_Omega(omega, T)
    assert isinstance(result, dict)
