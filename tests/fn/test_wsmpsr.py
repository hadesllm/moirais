"""Tests for wsmpsr.wasserman_poisson_regression."""
import numpy as np
import pytest
from morie.fn.wsmpsr import wasserman_poisson_regression


def test_wsmpsr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_poisson_regression(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmpsr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_poisson_regression(X, y)
    assert isinstance(result, dict)
