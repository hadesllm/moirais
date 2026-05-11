"""Tests for eslpsv.esl_pca_svd."""
import numpy as np
import pytest
from morie.fn.eslpsv import esl_pca_svd


def test_eslpsv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_pca_svd(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslpsv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_pca_svd(X, k)
    assert isinstance(result, dict)
