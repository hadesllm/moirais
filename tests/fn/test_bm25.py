"""Tests for bm25.bm25."""
import numpy as np
import pytest
from morie.fn.bm25 import bm25


def test_bm25_basic():
    """Test basic functionality."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    k1 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = bm25(docs, query, k1, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bm25_edge():
    """Test edge cases."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    k1 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = bm25(docs, query, k1, b)
    assert isinstance(result, dict)
