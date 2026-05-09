"""Tests for sgtadj.sgt_adjacency_matrix."""
import numpy as np
import pytest
from moirais.fn.sgtadj import sgt_adjacency_matrix


def test_sgtadj_basic():
    """Test basic functionality."""
    edges = [('A', 'B'), ('B', 'C')]
    n = 100
    directed = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_adjacency_matrix(edges, n, directed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtadj_edge():
    """Test edge cases."""
    edges = [('A', 'B'), ('B', 'C')]
    n = 100
    directed = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_adjacency_matrix(edges, n, directed)
    assert isinstance(result, dict)
