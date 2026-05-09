"""Tests for robpca.robust_pca."""
import numpy as np
import pytest
from moirais.fn.robpca import robust_pca


def test_robpca_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = robust_pca(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_robpca_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = robust_pca(X, k)
    assert isinstance(result, dict)
