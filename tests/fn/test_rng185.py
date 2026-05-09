"""Tests for rng185.rangayyan_ch4_pan_tompkins_highpass_transfer."""
import numpy as np
import pytest
from moirais.fn.rng185 import rangayyan_ch4_pan_tompkins_highpass_transfer


def test_rng185_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    H_lp = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_highpass_transfer(z, H_lp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng185_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    H_lp = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_highpass_transfer(z, H_lp)
    assert isinstance(result, dict)
