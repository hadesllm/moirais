"""Tests for sgtsck.sgt_spectral_clustering_k."""
import numpy as np
import pytest
from moirais.fn.sgtsck import sgt_spectral_clustering_k


def test_sgtsck_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = sgt_spectral_clustering_k(A, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtsck_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = sgt_spectral_clustering_k(A, k)
    assert isinstance(result, dict)
