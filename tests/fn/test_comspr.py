"""Tests for comspr.spectral_clustering."""
import numpy as np
import pytest
from moirais.fn.comspr import spectral_clustering


def test_comspr_basic():
    """Test basic functionality."""
    G = np.eye(10)
    k = 5
    result = spectral_clustering(G, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_comspr_edge():
    """Test edge cases."""
    G = np.eye(10)
    k = 5
    result = spectral_clustering(G, k)
    assert isinstance(result, dict)
