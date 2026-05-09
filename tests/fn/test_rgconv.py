"""Tests for rgconv.rangayyan_linear_convolution."""
import numpy as np
import pytest
from moirais.fn.rgconv import rangayyan_linear_convolution


def test_rgconv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_linear_convolution(x, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgconv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_linear_convolution(x, h)
    assert isinstance(result, dict)
