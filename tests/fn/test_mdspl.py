"""Tests for mdspl.mds_spatial_map."""
import numpy as np
import pytest
from morie.fn.mdspl import mds_spatial_map


def test_mdspl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mds_spatial_map(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mdspl_edge():
    """Test edge cases."""
    result = mds_spatial_map(np.array([42.0]))
    assert result['n'] == 1
