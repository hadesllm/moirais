"""Tests for alhds.alammar_hdbscan_cluster."""
import numpy as np
import pytest
from moirais.fn.alhds import alammar_hdbscan_cluster


def test_alhds_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    min_cluster_size = 100
    min_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_hdbscan_cluster(X, min_cluster_size, min_samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alhds_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    min_cluster_size = 100
    min_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_hdbscan_cluster(X, min_cluster_size, min_samples)
    assert isinstance(result, dict)
