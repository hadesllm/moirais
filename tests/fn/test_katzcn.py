"""Tests for katzcn.katz_centrality."""
import numpy as np
import pytest
from moirais.fn.katzcn import katz_centrality


def test_katzcn_basic():
    """Test basic functionality."""
    G = np.eye(10)
    alpha = 0.05
    beta = 0.8
    result = katz_centrality(G, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_katzcn_edge():
    """Test edge cases."""
    G = np.eye(10)
    alpha = 0.05
    beta = 0.8
    result = katz_centrality(G, alpha, beta)
    assert isinstance(result, dict)
