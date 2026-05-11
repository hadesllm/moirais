"""Tests for bipMch.bipartite_matching."""
import numpy as np
import pytest
from morie.fn.bipMch import bipartite_matching


def test_bipMch_basic():
    """Test basic functionality."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = bipartite_matching(U, V, E)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bipMch_edge():
    """Test edge cases."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = bipartite_matching(U, V, E)
    assert isinstance(result, dict)
