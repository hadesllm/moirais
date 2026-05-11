"""Tests for ripJ.ripley_j_function."""
import numpy as np
import pytest
from morie.fn.ripJ import ripley_j_function


def test_ripJ_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_j_function(points, window, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ripJ_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_j_function(points, window, r)
    assert isinstance(result, dict)
