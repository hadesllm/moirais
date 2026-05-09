"""Tests for hmadab.geron_adaboost."""
import numpy as np
import pytest
from moirais.fn.hmadab import geron_adaboost


def test_hmadab_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adaboost(X, y, base_estimator, n_estimators)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmadab_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adaboost(X, y, base_estimator, n_estimators)
    assert isinstance(result, dict)
