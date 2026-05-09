"""Tests for betcen.betweenness_centrality."""
import numpy as np
import pytest
from moirais.fn.betcen import betweenness_centrality


def test_betcen_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = betweenness_centrality(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_betcen_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = betweenness_centrality(G)
    assert isinstance(result, dict)
