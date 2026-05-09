"""Tests for eigcen.eigenvector_centrality."""
import numpy as np
import pytest
from moirais.fn.eigcen import eigenvector_centrality


def test_eigcen_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = eigenvector_centrality(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eigcen_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = eigenvector_centrality(G)
    assert isinstance(result, dict)
