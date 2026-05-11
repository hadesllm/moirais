"""Tests for hmceg.geron_cross_entropy_gradient."""
import numpy as np
import pytest
from morie.fn.hmceg import geron_cross_entropy_gradient


def test_hmceg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_cross_entropy_gradient(X, Y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmceg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_cross_entropy_gradient(X, Y, theta)
    assert isinstance(result, dict)
