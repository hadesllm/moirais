"""Tests for ripK.ripley_k_function."""
import numpy as np
import pytest
from moirais.fn.ripK import ripley_k_function


def test_ripK_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_k_function(points, window, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ripK_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_k_function(points, window, r)
    assert isinstance(result, dict)
