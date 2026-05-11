"""Tests for hmrfc.geron_random_forest."""
import numpy as np
import pytest
from morie.fn.hmrfc import geron_random_forest


def test_hmrfc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    max_features = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_random_forest(X, y, n_estimators, max_features, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrfc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    max_features = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_random_forest(X, y, n_estimators, max_features, seed)
    assert isinstance(result, dict)
