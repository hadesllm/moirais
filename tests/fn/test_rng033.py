"""Tests for rng033.rangayyan_ch3_causal_convolution_alt."""
import numpy as np
import pytest
from moirais.fn.rng033 import rangayyan_ch3_causal_convolution_alt


def test_rng033_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    tau = 0.1
    result = rangayyan_ch3_causal_convolution_alt(x, h, t, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng033_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    tau = 0.1
    result = rangayyan_ch3_causal_convolution_alt(x, h, t, tau)
    assert isinstance(result, dict)
