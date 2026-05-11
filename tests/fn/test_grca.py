"""Tests for grca.geron_cross_attention."""
import numpy as np
import pytest
from morie.fn.grca import geron_cross_attention


def test_grca_basic():
    """Test basic functionality."""
    X_dec = np.random.default_rng(42).normal(0, 1, 100)
    X_enc = np.random.default_rng(42).normal(0, 1, 100)
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_cross_attention(X_dec, X_enc, WQ, WK, WV)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grca_edge():
    """Test edge cases."""
    X_dec = np.random.default_rng(42).normal(0, 1, 100)
    X_enc = np.random.default_rng(42).normal(0, 1, 100)
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_cross_attention(X_dec, X_enc, WQ, WK, WV)
    assert isinstance(result, dict)
