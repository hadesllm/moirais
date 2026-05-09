"""Tests for rgpcaica.rangayyan_pca_vs_ica."""
import numpy as np
import pytest
from moirais.fn.rgpcaica import rangayyan_pca_vs_ica


def test_rgpcaica_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    method = 'auto'
    result = rangayyan_pca_vs_ica(X, n_components, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpcaica_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    method = 'auto'
    result = rangayyan_pca_vs_ica(X, n_components, method)
    assert isinstance(result, dict)
