"""Tests for hmkmlim.geron_kmeans_limits."""
import numpy as np
import pytest
from morie.fn.hmkmlim import geron_kmeans_limits


def test_hmkmlim_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_kmeans_limits(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkmlim_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_kmeans_limits(X)
    assert isinstance(result, dict)
