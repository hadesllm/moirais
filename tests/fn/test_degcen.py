"""Tests for degcen.degree_centrality."""
import numpy as np
import pytest
from moirais.fn.degcen import degree_centrality


def test_degcen_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = degree_centrality(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_degcen_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = degree_centrality(G)
    assert isinstance(result, dict)
