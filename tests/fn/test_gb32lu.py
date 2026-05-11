"""Tests for gb32lu.gibbons_runs_up_down_recur."""
import numpy as np
import pytest
from morie.fn.gb32lu import gibbons_runs_up_down_recur


def test_gb32lu_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_runs_up_down_recur(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb32lu_edge():
    """Test edge cases."""
    result = gibbons_runs_up_down_recur(np.array([42.0]))
    assert result['n'] == 1
