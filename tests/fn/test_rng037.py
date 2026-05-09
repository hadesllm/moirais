"""Tests for rng037.rangayyan_ch3_discrete_convolution_causal_alt."""
import numpy as np
import pytest
from moirais.fn.rng037 import rangayyan_ch3_discrete_convolution_causal_alt


def test_rng037_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    n = 100
    result = rangayyan_ch3_discrete_convolution_causal_alt(x, h, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng037_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    n = 100
    result = rangayyan_ch3_discrete_convolution_causal_alt(x, h, n)
    assert isinstance(result, dict)
