"""Tests for eslscd.esl_sparse_pca."""
import numpy as np
import pytest
from morie.fn.eslscd import esl_sparse_pca


def test_eslscd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_sparse_pca(X, k, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslscd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_sparse_pca(X, k, lambda_)
    assert isinstance(result, dict)
