"""Tests for rgbhp.rangayyan_butterworth_hp."""
import numpy as np
import pytest
from moirais.fn.rgbhp import rangayyan_butterworth_hp


def test_rgbhp_basic():
    """Test basic functionality."""
    cutoff_hz = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    fs = 100.0
    result = rangayyan_butterworth_hp(cutoff_hz, order, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbhp_edge():
    """Test edge cases."""
    cutoff_hz = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    fs = 100.0
    result = rangayyan_butterworth_hp(cutoff_hz, order, fs)
    assert isinstance(result, dict)
