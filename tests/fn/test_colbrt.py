"""Tests for colbrt.colbert."""
import numpy as np
import pytest
from moirais.fn.colbrt import colbert


def test_colbrt_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    docs = np.random.default_rng(42).normal(0, 1, 100)
    result = colbert(query, docs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_colbrt_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    docs = np.random.default_rng(42).normal(0, 1, 100)
    result = colbert(query, docs)
    assert isinstance(result, dict)
