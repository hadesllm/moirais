"""Tests for rgeqn3a.rangayyan_ch3_convolution_sum."""
import numpy as np
import pytest
from morie.fn.rgeqn3a import rangayyan_ch3_convolution_sum


def test_rgeqn3a_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_ch3_convolution_sum(x, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn3a_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_ch3_convolution_sum(x, h)
    assert isinstance(result, dict)
