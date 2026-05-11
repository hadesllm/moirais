"""Tests for rfmlt.rf_multivariate."""
import numpy as np
import pytest
from morie.fn.rfmlt import rf_multivariate


def test_rfmlt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    result = rf_multivariate(X, Y_matrix, n_trees)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rfmlt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    result = rf_multivariate(X, Y_matrix, n_trees)
    assert isinstance(result, dict)
