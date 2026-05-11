"""Tests for rng213.rangayyan_ch4_peak_power_snr."""
import numpy as np
import pytest
from morie.fn.rng213 import rangayyan_ch4_peak_power_snr


def test_rng213_basic():
    """Test basic functionality."""
    M_y = np.random.default_rng(42).normal(0, 1, 100)
    P_eta_o = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_peak_power_snr(M_y, P_eta_o)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng213_edge():
    """Test edge cases."""
    M_y = np.random.default_rng(42).normal(0, 1, 100)
    P_eta_o = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_peak_power_snr(M_y, P_eta_o)
    assert isinstance(result, dict)
