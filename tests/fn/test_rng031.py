"""Tests for rng031.rangayyan_ch3_continuous_convolution_alt."""
import numpy as np
import pytest
from morie.fn.rng031 import rangayyan_ch3_continuous_convolution_alt


def test_rng031_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    tau = 0.1
    result = rangayyan_ch3_continuous_convolution_alt(x, h, t, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng031_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    tau = 0.1
    result = rangayyan_ch3_continuous_convolution_alt(x, h, t, tau)
    assert isinstance(result, dict)
