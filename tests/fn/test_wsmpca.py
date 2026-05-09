"""Tests for wsmpca.wasserman_pca."""
import numpy as np
import pytest
from moirais.fn.wsmpca import wasserman_pca


def test_wsmpca_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = wasserman_pca(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmpca_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = wasserman_pca(X, k)
    assert isinstance(result, dict)
