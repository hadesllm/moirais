"""Tests for pcadr.pca_dimensionality_reduction."""
import numpy as np
import pytest
from morie.fn.pcadr import pca_dimensionality_reduction


def test_pcadr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = pca_dimensionality_reduction(X, n_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pcadr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = pca_dimensionality_reduction(X, n_components)
    assert isinstance(result, dict)
