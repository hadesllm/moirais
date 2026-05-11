"""Tests for ripL.ripley_l_function."""
import numpy as np
import pytest
from morie.fn.ripL import ripley_l_function


def test_ripL_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_l_function(points, window, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ripL_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_l_function(points, window, r)
    assert isinstance(result, dict)
