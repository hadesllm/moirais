"""Tests for reglmd.regression_estimator."""
import numpy as np
import pytest
from morie.fn.reglmd import regression_estimator


def test_reglmd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = regression_estimator(y, x, X, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reglmd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = regression_estimator(y, x, X, weights)
    assert isinstance(result, dict)
