"""Tests for wsmdir.wasserman_directed_graph."""
import numpy as np
import pytest
from morie.fn.wsmdir import wasserman_directed_graph


def test_wsmdir_basic():
    """Test basic functionality."""
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_directed_graph(dag, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmdir_edge():
    """Test edge cases."""
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_directed_graph(dag, x)
    assert isinstance(result, dict)
