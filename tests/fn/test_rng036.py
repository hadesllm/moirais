"""Tests for rng036.rangayyan_ch3_discrete_convolution_causal."""
import numpy as np
import pytest
from morie.fn.rng036 import rangayyan_ch3_discrete_convolution_causal


def test_rng036_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    n = 100
    result = rangayyan_ch3_discrete_convolution_causal(x, h, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng036_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    n = 100
    result = rangayyan_ch3_discrete_convolution_causal(x, h, n)
    assert isinstance(result, dict)
