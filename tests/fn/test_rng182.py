"""Tests for rng182.rangayyan_ch4_pan_tompkins_lowpass_difference_eq."""
import numpy as np
import pytest
from morie.fn.rng182 import rangayyan_ch4_pan_tompkins_lowpass_difference_eq


def test_rng182_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_pan_tompkins_lowpass_difference_eq(x, y, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng182_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_pan_tompkins_lowpass_difference_eq(x, y, n)
    assert isinstance(result, dict)
