"""Tests for hmrsp.geron_random_subspaces."""
import numpy as np
import pytest
from morie.fn.hmrsp import geron_random_subspaces


def test_hmrsp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    max_features = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_random_subspaces(X, y, base_estimator, n_estimators, max_features)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrsp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    max_features = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_random_subspaces(X, y, base_estimator, n_estimators, max_features)
    assert isinstance(result, dict)
