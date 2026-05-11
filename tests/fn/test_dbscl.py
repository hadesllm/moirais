"""Tests for dbscl.dbscan_clustering."""
import numpy as np
import pytest
from morie.fn.dbscl import dbscan_clustering


def test_dbscl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = dbscan_clustering(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_dbscl_edge():
    """Test edge cases."""
    result = dbscan_clustering(np.array([42.0]))
    assert result['n'] == 1
