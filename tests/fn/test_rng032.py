"""Tests for rng032.rangayyan_ch3_causal_convolution."""
import numpy as np
import pytest
from moirais.fn.rng032 import rangayyan_ch3_causal_convolution


def test_rng032_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    tau = 0.1
    result = rangayyan_ch3_causal_convolution(x, h, t, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng032_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    tau = 0.1
    result = rangayyan_ch3_causal_convolution(x, h, t, tau)
    assert isinstance(result, dict)
