"""Tests for brdgr.bridge_observations."""
import numpy as np
import pytest
from morie.fn.brdgr import bridge_observations


def test_brdgr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bridge_observations(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_brdgr_edge():
    """Test edge cases."""
    result = bridge_observations(np.array([42.0]))
    assert result['n'] == 1
