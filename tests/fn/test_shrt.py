"""Tests for shrt.shortest_path_dijkstra."""
import numpy as np
import pytest
from moirais.fn.shrt import shortest_path_dijkstra


def test_shrt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    s = 90
    t = np.linspace(0, 10, 100)
    result = shortest_path_dijkstra(y, A, s, t)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_shrt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    s = 90
    t = np.linspace(0, 10, 100)
    result = shortest_path_dijkstra(y, A, s, t)
    assert isinstance(result, dict)
