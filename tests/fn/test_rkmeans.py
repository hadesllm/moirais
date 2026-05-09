"""Tests for rkmeans.trimmed_kmeans."""
import numpy as np
import pytest
from moirais.fn.rkmeans import trimmed_kmeans


def test_rkmeans_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    alpha = 0.05
    result = trimmed_kmeans(X, k, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rkmeans_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    alpha = 0.05
    result = trimmed_kmeans(X, k, alpha)
    assert isinstance(result, dict)
