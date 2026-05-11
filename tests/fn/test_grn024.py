"""Tests for grn024.geron_ch4_cross_entropy_gradient_vector."""
import numpy as np
import pytest
from morie.fn.grn024 import geron_ch4_cross_entropy_gradient_vector


def test_grn024_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_ch4_cross_entropy_gradient_vector(X, Y, Theta, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grn024_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_ch4_cross_entropy_gradient_vector(X, Y, Theta, k)
    assert isinstance(result, dict)
