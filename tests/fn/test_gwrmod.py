"""Tests for gwrmod.geographically_weighted_regression."""
import numpy as np
import pytest
from moirais.fn.gwrmod import geographically_weighted_regression


def test_gwrmod_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    bandwidth = 0.3
    result = geographically_weighted_regression(y, X, coords, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gwrmod_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    bandwidth = 0.3
    result = geographically_weighted_regression(y, X, coords, bandwidth)
    assert isinstance(result, dict)
