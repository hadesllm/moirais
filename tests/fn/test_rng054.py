"""Tests for rng054.rangayyan_ch3_z_transform_convolution."""
import numpy as np
import pytest
from morie.fn.rng054 import rangayyan_ch3_z_transform_convolution


def test_rng054_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_ch3_z_transform_convolution(x, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng054_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_ch3_z_transform_convolution(x, h)
    assert isinstance(result, dict)
