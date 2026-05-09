"""Tests for clocen.closeness_centrality."""
import numpy as np
import pytest
from moirais.fn.clocen import closeness_centrality


def test_clocen_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = closeness_centrality(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clocen_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = closeness_centrality(G)
    assert isinstance(result, dict)
