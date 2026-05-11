"""Tests for rghrvt.rangayyan_hrv_time_domain."""
import numpy as np
import pytest
from morie.fn.rghrvt import rangayyan_hrv_time_domain


def test_rghrvt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_hrv_time_domain(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rghrvt_edge():
    """Test edge cases."""
    result = rangayyan_hrv_time_domain(np.array([42.0]))
    assert result['n'] == 1
