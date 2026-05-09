"""Tests for gb2312.gibbons_edf_consistent."""
import numpy as np
import pytest
from moirais.fn.gb2312 import gibbons_edf_consistent


def test_gb2312_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_edf_consistent(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb2312_edge():
    """Test edge cases."""
    result = gibbons_edf_consistent(np.array([42.0]))
    assert result['n'] == 1
