"""Tests for rgarb.rangayyan_ar_burg."""
import numpy as np
import pytest
from morie.fn.rgarb import rangayyan_ar_burg


def test_rgarb_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_ar_burg(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgarb_edge():
    """Test edge cases."""
    result = rangayyan_ar_burg(np.array([42.0]))
    assert result['n'] == 1
