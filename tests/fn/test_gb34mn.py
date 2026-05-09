"""Tests for gb34mn.gibbons_runs_ud_mean."""
import numpy as np
import pytest
from moirais.fn.gb34mn import gibbons_runs_ud_mean


def test_gb34mn_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_runs_ud_mean(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb34mn_edge():
    """Test edge cases."""
    result = gibbons_runs_ud_mean(np.array([42.0]))
    assert result['n'] == 1
