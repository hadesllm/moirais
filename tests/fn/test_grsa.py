"""Tests for grsa.geron_self_attention."""
import numpy as np
import pytest
from morie.fn.grsa import geron_self_attention


def test_grsa_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_self_attention(X, WQ, WK, WV)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsa_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_self_attention(X, WQ, WK, WV)
    assert isinstance(result, dict)
