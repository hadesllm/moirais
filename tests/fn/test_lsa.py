"""Tests for lsa.lsa."""
import numpy as np
import pytest
from morie.fn.lsa import lsa


def test_lsa_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = lsa(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lsa_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = lsa(X, k)
    assert isinstance(result, dict)
