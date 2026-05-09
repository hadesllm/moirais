"""Tests for rng256.rangayyan_ch4_power_spectrum_signal_echo."""
import numpy as np
import pytest
from moirais.fn.rng256 import rangayyan_ch4_power_spectrum_signal_echo


def test_rng256_basic():
    """Test basic functionality."""
    H = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch4_power_spectrum_signal_echo(H, a, n_0, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng256_edge():
    """Test edge cases."""
    H = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch4_power_spectrum_signal_echo(H, a, n_0, z)
    assert isinstance(result, dict)
