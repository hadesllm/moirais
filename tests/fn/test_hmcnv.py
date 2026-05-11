"""Tests for hmcnv.geron_convolutional_layer."""
import numpy as np
import pytest
from morie.fn.hmcnv import geron_convolutional_layer


def test_hmcnv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    stride = np.random.default_rng(42).normal(0, 1, 100)
    padding = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_convolutional_layer(x, kernel, stride, padding)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hmcnv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    stride = np.random.default_rng(42).normal(0, 1, 100)
    padding = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_convolutional_layer(x, kernel, stride, padding)
    assert isinstance(result, dict)
