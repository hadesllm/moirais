"""Tests for kmnsc.kmeans_clustering."""
import numpy as np
import pytest
from morie.fn.kmnsc import kmeans_clustering


def test_kmnsc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kmeans_clustering(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_kmnsc_edge():
    """Test edge cases."""
    result = kmeans_clustering(np.array([42.0]))
    assert result['n'] == 1
