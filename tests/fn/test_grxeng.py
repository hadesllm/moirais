"""Tests for grxeng.geron_softmax_cost_gradient."""
import numpy as np
import pytest
from morie.fn.grxeng import geron_softmax_cost_gradient


def test_grxeng_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_softmax_cost_gradient(X, Y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grxeng_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_softmax_cost_gradient(X, Y, theta)
    assert isinstance(result, dict)
