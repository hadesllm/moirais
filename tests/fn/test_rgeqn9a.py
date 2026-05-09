"""Tests for rgeqn9a.rangayyan_ch9_pca_reconstruction."""
import numpy as np
import pytest
from moirais.fn.rgeqn9a import rangayyan_ch9_pca_reconstruction


def test_rgeqn9a_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = rangayyan_ch9_pca_reconstruction(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn9a_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = rangayyan_ch9_pca_reconstruction(X, k)
    assert isinstance(result, dict)
