"""Tests for rgeqn3b.rangayyan_ch3_correlation_sum."""
import numpy as np
import pytest
from moirais.fn.rgeqn3b import rangayyan_ch3_correlation_sum


def test_rgeqn3b_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = rangayyan_ch3_correlation_sum(x, y)
    assert abs(result['statistic'] - 1.0) < 0.01


def test_rgeqn3b_edge():
    """Test edge cases."""
    result = rangayyan_ch3_correlation_sum(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result['n'] == 2
