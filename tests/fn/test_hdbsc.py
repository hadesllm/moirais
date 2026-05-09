"""Tests for hdbsc.hdbscan."""
import numpy as np
import pytest
from moirais.fn.hdbsc import hdbscan


def test_hdbsc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    min_cluster_size = 100
    result = hdbscan(X, min_cluster_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hdbsc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    min_cluster_size = 100
    result = hdbscan(X, min_cluster_size)
    assert isinstance(result, dict)
