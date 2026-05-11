"""Tests for comten.community_modularity."""
import numpy as np
import pytest
from morie.fn.comten import community_modularity


def test_comten_basic():
    """Test basic functionality."""
    G = np.eye(10)
    partition = np.random.default_rng(42).normal(0, 1, 100)
    result = community_modularity(G, partition)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_comten_edge():
    """Test edge cases."""
    G = np.eye(10)
    partition = np.random.default_rng(42).normal(0, 1, 100)
    result = community_modularity(G, partition)
    assert isinstance(result, dict)
