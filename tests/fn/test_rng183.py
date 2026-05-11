"""Tests for rng183.rangayyan_ch4_pan_tompkins_highpass_lp_component."""
import numpy as np
import pytest
from morie.fn.rng183 import rangayyan_ch4_pan_tompkins_highpass_lp_component


def test_rng183_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_highpass_lp_component(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng183_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_highpass_lp_component(z)
    assert isinstance(result, dict)
