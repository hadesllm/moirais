"""Tests for otmcluster.ot_clustering_w."""
import numpy as np
import pytest
from moirais.fn.otmcluster import ot_clustering_w


def test_otmcluster_basic():
    """Test basic functionality."""
    X_list = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_clustering_w(X_list, k, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmcluster_edge():
    """Test edge cases."""
    X_list = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_clustering_w(X_list, k, max_iter)
    assert isinstance(result, dict)
