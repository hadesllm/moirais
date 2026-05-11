"""Tests for greast.geron_early_stopping."""
import numpy as np
import pytest
from morie.fn.greast import geron_early_stopping


def test_greast_basic():
    """Test basic functionality."""
    X_train = np.random.default_rng(42).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    X_val = np.random.default_rng(42).normal(0, 1, 100)
    y_val = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_early_stopping(X_train, y_train, X_val, y_val, n_iter, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_greast_edge():
    """Test edge cases."""
    X_train = np.random.default_rng(42).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    X_val = np.random.default_rng(42).normal(0, 1, 100)
    y_val = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_early_stopping(X_train, y_train, X_val, y_val, n_iter, eta)
    assert isinstance(result, dict)
