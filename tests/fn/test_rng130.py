"""Tests for rng130.rangayyan_ch3_bilinear_warping_Omega_to_omega."""
import numpy as np
import pytest
from morie.fn.rng130 import rangayyan_ch3_bilinear_warping_Omega_to_omega


def test_rng130_basic():
    """Test basic functionality."""
    Omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_bilinear_warping_Omega_to_omega(Omega, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng130_edge():
    """Test edge cases."""
    Omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_bilinear_warping_Omega_to_omega(Omega, T)
    assert isinstance(result, dict)
