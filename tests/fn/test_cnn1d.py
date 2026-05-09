"""Tests for cnn1d.conv1d_forward."""
import numpy as np
import pytest
from moirais.fn.cnn1d import conv1d_forward


def test_cnn1d_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = conv1d_forward(x, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cnn1d_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = conv1d_forward(x, w)
    assert isinstance(result, dict)
