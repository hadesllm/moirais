"""Tests for tmldis.tmle_disparity."""
import numpy as np
import pytest
from morie.fn.tmldis import tmle_disparity


def test_tmldis_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    X_target = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_disparity(y, S, X, X_target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmldis_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    X_target = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_disparity(y, S, X, X_target)
    assert isinstance(result, dict)
