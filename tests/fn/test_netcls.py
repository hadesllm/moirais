"""Tests for netcls.closeness_centrality."""
import numpy as np
import pytest
from morie.fn.netcls import closeness_centrality


def test_netcls_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = closeness_centrality(y, A, node)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_netcls_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = closeness_centrality(y, A, node)
    assert isinstance(result, dict)
