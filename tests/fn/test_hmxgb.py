"""Tests for hmxgb.geron_xgboost."""
import numpy as np
import pytest
from morie.fn.hmxgb import geron_xgboost


def test_hmxgb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    learning_rate = 0.1
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_xgboost(X, y, n_estimators, learning_rate, max_depth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmxgb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    learning_rate = 0.1
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_xgboost(X, y, n_estimators, learning_rate, max_depth)
    assert isinstance(result, dict)
