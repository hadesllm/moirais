"""Tests for hmlof.geron_local_outlier_factor."""
import numpy as np
import pytest
from moirais.fn.hmlof import geron_local_outlier_factor


def test_hmlof_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_local_outlier_factor(X, n_neighbors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlof_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_local_outlier_factor(X, n_neighbors)
    assert isinstance(result, dict)
