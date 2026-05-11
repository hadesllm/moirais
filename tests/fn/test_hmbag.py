"""Tests for hmbag.geron_bagging."""
import numpy as np
import pytest
from morie.fn.hmbag import geron_bagging


def test_hmbag_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_bagging(X, y, base_estimator, n_estimators, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbag_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_bagging(X, y, base_estimator, n_estimators, seed)
    assert isinstance(result, dict)
