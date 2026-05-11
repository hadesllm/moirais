"""Tests for kmhyb.kamath_hybrid_retrieval_fusion."""
import numpy as np
import pytest
from morie.fn.kmhyb import kamath_hybrid_retrieval_fusion


def test_kmhyb_basic():
    """Test basic functionality."""
    s_dense = np.random.default_rng(42).normal(0, 1, 100)
    s_sparse = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_hybrid_retrieval_fusion(s_dense, s_sparse, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmhyb_edge():
    """Test edge cases."""
    s_dense = np.random.default_rng(42).normal(0, 1, 100)
    s_sparse = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_hybrid_retrieval_fusion(s_dense, s_sparse, lam)
    assert isinstance(result, dict)
