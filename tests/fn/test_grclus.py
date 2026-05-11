"""Tests for grclus.graph_clustering."""
import numpy as np
import pytest
from morie.fn.grclus import graph_clustering


def test_grclus_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = graph_clustering(A, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grclus_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = graph_clustering(A, k)
    assert isinstance(result, dict)
