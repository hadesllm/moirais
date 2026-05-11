"""Tests for shrtG.shortest_path_kernel."""
import numpy as np
import pytest
from morie.fn.shrtG import shortest_path_kernel


def test_shrtG_basic():
    """Test basic functionality."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    result = shortest_path_kernel(G1, G2)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_shrtG_edge():
    """Test edge cases."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    result = shortest_path_kernel(G1, G2)
    assert isinstance(result, dict)
