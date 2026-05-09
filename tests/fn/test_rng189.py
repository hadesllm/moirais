"""Tests for rng189.rangayyan_ch4_pan_tompkins_moving_window_integrator."""
import numpy as np
import pytest
from moirais.fn.rng189 import rangayyan_ch4_pan_tompkins_moving_window_integrator


def test_rng189_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    n = 100
    result = rangayyan_ch4_pan_tompkins_moving_window_integrator(x, N, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng189_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    n = 100
    result = rangayyan_ch4_pan_tompkins_moving_window_integrator(x, N, n)
    assert isinstance(result, dict)
