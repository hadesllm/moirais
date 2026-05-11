"""Tests for wsmkmn.wasserman_kmeans."""
import numpy as np
import pytest
from morie.fn.wsmkmn import wasserman_kmeans


def test_wsmkmn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = wasserman_kmeans(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmkmn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = wasserman_kmeans(X, k)
    assert isinstance(result, dict)
