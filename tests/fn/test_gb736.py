"""Tests for gb736.gibbons_linrank_sym_special."""
import numpy as np
import pytest
from morie.fn.gb736 import gibbons_linrank_sym_special


def test_gb736_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_linrank_sym_special(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb736_edge():
    """Test edge cases."""
    result = gibbons_linrank_sym_special(np.array([42.0]))
    assert result['n'] == 1
