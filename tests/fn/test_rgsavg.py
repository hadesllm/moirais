"""Tests for rgsavg.rangayyan_sync_average."""
import numpy as np
import pytest
from morie.fn.rgsavg import rangayyan_sync_average


def test_rgsavg_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_sync_average(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgsavg_edge():
    """Test edge cases."""
    result = rangayyan_sync_average(np.array([42.0]))
    assert result['n'] == 1
