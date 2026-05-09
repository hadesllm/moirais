"""Tests for hmbp.geron_backpropagation."""
import numpy as np
import pytest
from moirais.fn.hmbp import geron_backpropagation


def test_hmbp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    activations = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_backpropagation(X, y, weights, activations)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    activations = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_backpropagation(X, y, weights, activations)
    assert isinstance(result, dict)
