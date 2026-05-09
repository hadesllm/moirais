"""Tests for rghfd.rangayyan_higuchi_fd."""
import numpy as np
import pytest
from moirais.fn.rghfd import rangayyan_higuchi_fd


def test_rghfd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_higuchi_fd(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rghfd_edge():
    """Test edge cases."""
    result = rangayyan_higuchi_fd(np.array([42.0]))
    assert result['n'] == 1
