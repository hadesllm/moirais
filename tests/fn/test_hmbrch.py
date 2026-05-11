"""Tests for hmbrch.geron_birch."""
import numpy as np
import pytest
from morie.fn.hmbrch import geron_birch


def test_hmbrch_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    branching_factor = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_birch(X, n_clusters, threshold, branching_factor)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbrch_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    branching_factor = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_birch(X, n_clusters, threshold, branching_factor)
    assert isinstance(result, dict)
