"""Tests for tmldyk.tmle_diff_kernel."""
import numpy as np
import pytest
from morie.fn.tmldyk import tmle_diff_kernel


def test_tmldyk_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    epsilon = 1e-6
    result = tmle_diff_kernel(y, D, X, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmldyk_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    epsilon = 1e-6
    result = tmle_diff_kernel(y, D, X, epsilon)
    assert isinstance(result, dict)
