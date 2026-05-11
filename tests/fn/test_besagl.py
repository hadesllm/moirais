"""Tests for besagl.besag_York_Mollie."""
import numpy as np
import pytest
from morie.fn.besagl import besag_York_Mollie


def test_besagl_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    offset = np.random.default_rng(42).normal(0, 1, 100)
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    result = besag_York_Mollie(counts, X, offset, adjacency)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_besagl_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    offset = np.random.default_rng(42).normal(0, 1, 100)
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    result = besag_York_Mollie(counts, X, offset, adjacency)
    assert isinstance(result, dict)
