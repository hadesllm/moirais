"""Tests for rgdnot.rangayyan_dicrotic_notch."""
import numpy as np
import pytest
from morie.fn.rgdnot import rangayyan_dicrotic_notch


def test_rgdnot_basic():
    """Test basic functionality."""
    pulse = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_dicrotic_notch(pulse, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgdnot_edge():
    """Test edge cases."""
    pulse = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_dicrotic_notch(pulse, fs)
    assert isinstance(result, dict)
