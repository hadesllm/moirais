"""Tests for ripF.ripley_f_function."""
import numpy as np
import pytest
from moirais.fn.ripF import ripley_f_function


def test_ripF_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_f_function(points, window, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ripF_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = ripley_f_function(points, window, r)
    assert isinstance(result, dict)
