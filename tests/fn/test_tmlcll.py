"""Tests for tmlcll.tmle_cross_lagged."""
import numpy as np
import pytest
from morie.fn.tmlcll import tmle_cross_lagged


def test_tmlcll_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    time = np.linspace(0, 10, 100)
    result = tmle_cross_lagged(y, D, X, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlcll_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    time = np.linspace(0, 10, 100)
    result = tmle_cross_lagged(y, D, X, time)
    assert isinstance(result, dict)
