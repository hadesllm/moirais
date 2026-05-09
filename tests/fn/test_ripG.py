"""Tests for ripG.ripley_g_function."""
import numpy as np
import pytest
from moirais.fn.ripG import ripley_g_function


def test_ripG_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_g_function(points, window, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ripG_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_g_function(points, window, r)
    assert isinstance(result, dict)
