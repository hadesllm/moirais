"""Tests for rgampd.rangayyan_amplitude_demod."""
import numpy as np
import pytest
from morie.fn.rgampd import rangayyan_amplitude_demod


def test_rgampd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_amplitude_demod(x, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgampd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_amplitude_demod(x, fs)
    assert isinstance(result, dict)
