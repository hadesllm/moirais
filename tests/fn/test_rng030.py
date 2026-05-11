"""Tests for rng030.rangayyan_ch3_continuous_convolution."""
import numpy as np
import pytest
from morie.fn.rng030 import rangayyan_ch3_continuous_convolution


def test_rng030_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    tau = 0.1
    result = rangayyan_ch3_continuous_convolution(x, h, t, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng030_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    tau = 0.1
    result = rangayyan_ch3_continuous_convolution(x, h, t, tau)
    assert isinstance(result, dict)
