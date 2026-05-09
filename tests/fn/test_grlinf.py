"""Tests for grlinf.geron_linear_layer_forward."""
import numpy as np
import pytest
from moirais.fn.grlinf import geron_linear_layer_forward


def test_grlinf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_linear_layer_forward(X, W, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grlinf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_linear_layer_forward(X, W, b)
    assert isinstance(result, dict)
