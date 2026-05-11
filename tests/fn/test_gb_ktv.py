"""Tests for gb_ktv.gibbons_kendall_tau_var."""
import numpy as np
import pytest
from morie.fn.gb_ktv import gibbons_kendall_tau_var


def test_gb_ktv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_kendall_tau_var(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_ktv_edge():
    """Test edge cases."""
    result = gibbons_kendall_tau_var(np.array([42.0]))
    assert result['n'] == 1
