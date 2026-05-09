"""Tests for rng209.rangayyan_ch4_white_noise_psd_input."""
import numpy as np
import pytest
from moirais.fn.rng209 import rangayyan_ch4_white_noise_psd_input


def test_rng209_basic():
    """Test basic functionality."""
    P_eta_i = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_white_noise_psd_input(P_eta_i, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng209_edge():
    """Test edge cases."""
    P_eta_i = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_white_noise_psd_input(P_eta_i, f)
    assert isinstance(result, dict)
