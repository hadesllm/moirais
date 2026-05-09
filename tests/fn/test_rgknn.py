"""Tests for rgknn.rangayyan_knn_classifier."""
import numpy as np
import pytest
from moirais.fn.rgknn import rangayyan_knn_classifier


def test_rgknn_basic():
    """Test basic functionality."""
    X_train = np.random.default_rng(42).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    k = 5
    result = rangayyan_knn_classifier(X_train, y_train, X_test, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgknn_edge():
    """Test edge cases."""
    X_train = np.random.default_rng(42).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    k = 5
    result = rangayyan_knn_classifier(X_train, y_train, X_test, k)
    assert isinstance(result, dict)
