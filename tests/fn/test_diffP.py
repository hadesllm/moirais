"""Tests for diffP.diffpool."""
import numpy as np
import pytest
from moirais.fn.diffP import diffpool


def test_diffP_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K_clusters = np.random.default_rng(42).normal(0, 1, 100)
    result = diffpool(A, X, K_clusters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diffP_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K_clusters = np.random.default_rng(42).normal(0, 1, 100)
    result = diffpool(A, X, K_clusters)
    assert isinstance(result, dict)
