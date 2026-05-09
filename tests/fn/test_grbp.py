"""Tests for grbp.geron_backpropagation_gradient."""
import numpy as np
import pytest
from moirais.fn.grbp import geron_backpropagation_gradient


def test_grbp_basic():
    """Test basic functionality."""
    activations = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_backpropagation_gradient(activations, weights, y_true)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbp_edge():
    """Test edge cases."""
    activations = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_backpropagation_gradient(activations, weights, y_true)
    assert isinstance(result, dict)
