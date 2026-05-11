"""Tests for grcvf.geron_conv2d_forward."""
import numpy as np
import pytest
from morie.fn.grcvf import geron_conv2d_forward


def test_grcvf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    padding = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_conv2d_forward(X, W, b, stride, padding)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grcvf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    padding = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_conv2d_forward(X, W, b, stride, padding)
    assert isinstance(result, dict)
