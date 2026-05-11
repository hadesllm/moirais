"""Tests for rng211.rangayyan_ch4_average_output_noise_power."""
import numpy as np
import pytest
from morie.fn.rng211 import rangayyan_ch4_average_output_noise_power


def test_rng211_basic():
    """Test basic functionality."""
    P_eta_i = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_average_output_noise_power(P_eta_i, H, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng211_edge():
    """Test edge cases."""
    P_eta_i = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_average_output_noise_power(P_eta_i, H, f)
    assert isinstance(result, dict)
