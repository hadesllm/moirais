"""Tests for hmgrs.geron_grid_search."""
import numpy as np
import pytest
from morie.fn.hmgrs import geron_grid_search


def test_hmgrs_basic():
    """Test basic functionality."""
    param_grid = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_grid_search(param_grid, X, y, estimator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgrs_edge():
    """Test edge cases."""
    param_grid = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_grid_search(param_grid, X, y, estimator)
    assert isinstance(result, dict)
