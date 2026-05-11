"""Tests for hmext.geron_extra_trees."""
import numpy as np
import pytest
from morie.fn.hmext import geron_extra_trees


def test_hmext_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    max_features = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_extra_trees(X, y, n_estimators, max_features, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmext_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    max_features = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_extra_trees(X, y, n_estimators, max_features, seed)
    assert isinstance(result, dict)
