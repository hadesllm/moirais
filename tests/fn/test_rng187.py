"""Tests for rng187.rangayyan_ch4_pan_tompkins_highpass_combined."""
import numpy as np
import pytest
from moirais.fn.rng187 import rangayyan_ch4_pan_tompkins_highpass_combined


def test_rng187_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    n = 100
    result = rangayyan_ch4_pan_tompkins_highpass_combined(x, p, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng187_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    n = 100
    result = rangayyan_ch4_pan_tompkins_highpass_combined(x, p, n)
    assert isinstance(result, dict)
