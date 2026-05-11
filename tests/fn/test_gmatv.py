"""Tests for gmatv.grm_vanraden."""
import numpy as np
import pytest
from morie.fn.gmatv import grm_vanraden


def test_gmatv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = grm_vanraden(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gmatv_edge():
    """Test edge cases."""
    result = grm_vanraden(np.array([42.0]))
    assert result['n'] == 1
