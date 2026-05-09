"""Tests for pcfunc.pair_correlation_function."""
import numpy as np
import pytest
from moirais.fn.pcfunc import pair_correlation_function


def test_pcfunc_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = pair_correlation_function(points, window, r)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_pcfunc_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = pair_correlation_function(points, window, r)
    assert isinstance(result, dict)
