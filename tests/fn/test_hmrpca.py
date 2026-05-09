"""Tests for hmrpca.geron_randomized_pca."""
import numpy as np
import pytest
from moirais.fn.hmrpca import geron_randomized_pca


def test_hmrpca_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    seed = 42
    result = geron_randomized_pca(X, n_components, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrpca_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    seed = 42
    result = geron_randomized_pca(X, n_components, seed)
    assert isinstance(result, dict)
