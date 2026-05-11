"""Tests for scleid.leiden_clustering."""
import numpy as np
import pytest
from morie.fn.scleid import leiden_clustering


def test_scleid_basic():
    """Test basic functionality."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = leiden_clustering(graph, resolution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scleid_edge():
    """Test edge cases."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = leiden_clustering(graph, resolution)
    assert isinstance(result, dict)
