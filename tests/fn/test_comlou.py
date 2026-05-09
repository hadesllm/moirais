"""Tests for comlou.louvain_communities."""
import numpy as np
import pytest
from moirais.fn.comlou import louvain_communities


def test_comlou_basic():
    """Test basic functionality."""
    G = np.eye(10)
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = louvain_communities(G, resolution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_comlou_edge():
    """Test edge cases."""
    G = np.eye(10)
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = louvain_communities(G, resolution)
    assert isinstance(result, dict)
