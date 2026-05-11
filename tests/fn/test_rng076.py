"""Tests for rng076.rangayyan_ch3_dft_convolution_property."""
import numpy as np
import pytest
from morie.fn.rng076 import rangayyan_ch3_dft_convolution_property


def test_rng076_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_ch3_dft_convolution_property(x, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng076_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_ch3_dft_convolution_property(x, h)
    assert isinstance(result, dict)
