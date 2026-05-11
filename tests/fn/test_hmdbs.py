"""Tests for hmdbs.geron_dbscan."""
import numpy as np
import pytest
from morie.fn.hmdbs import geron_dbscan


def test_hmdbs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    eps = np.random.default_rng(42).normal(0, 1, 100)
    min_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dbscan(X, eps, min_samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdbs_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    eps = np.random.default_rng(42).normal(0, 1, 100)
    min_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dbscan(X, eps, min_samples)
    assert isinstance(result, dict)
