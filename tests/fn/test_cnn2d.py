"""Tests for cnn2d.conv2d_forward."""
import numpy as np
import pytest
from morie.fn.cnn2d import conv2d_forward


def test_cnn2d_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = conv2d_forward(x, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cnn2d_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = conv2d_forward(x, w)
    assert isinstance(result, dict)
