"""Tests for grgs.geron_grid_search_cv."""
import numpy as np
import pytest
from moirais.fn.grgs import geron_grid_search_cv


def test_grgs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    param_grid = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = geron_grid_search_cv(X, y, param_grid, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgs_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    param_grid = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = geron_grid_search_cv(X, y, param_grid, K)
    assert isinstance(result, dict)
