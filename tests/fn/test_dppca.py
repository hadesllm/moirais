"""Tests for dppca.dp_pca."""
import numpy as np
import pytest
from morie.fn.dppca import dp_pca


def test_dppca_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    epsilon = 1e-6
    result = dp_pca(X, k, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dppca_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    epsilon = 1e-6
    result = dp_pca(X, k, epsilon)
    assert isinstance(result, dict)
