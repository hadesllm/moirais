"""Tests for hmsslc.geron_semisupervised_cluster."""
import numpy as np
import pytest
from morie.fn.hmsslc import geron_semisupervised_cluster


def test_hmsslc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    X_labeled = np.random.default_rng(42).normal(0, 1, 100)
    y_labeled = np.random.default_rng(42).normal(0, 1, 100)
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_semisupervised_cluster(X, X_labeled, y_labeled, n_clusters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsslc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    X_labeled = np.random.default_rng(42).normal(0, 1, 100)
    y_labeled = np.random.default_rng(42).normal(0, 1, 100)
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_semisupervised_cluster(X, X_labeled, y_labeled, n_clusters)
    assert isinstance(result, dict)
