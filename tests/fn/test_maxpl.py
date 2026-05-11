"""Tests for maxpl.max_pooling."""
import numpy as np
import pytest
from morie.fn.maxpl import max_pooling


def test_maxpl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = max_pooling(x, kernel, stride)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_maxpl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = max_pooling(x, kernel, stride)
    assert isinstance(result, dict)
