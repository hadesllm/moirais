"""Tests for rgcrl.rangayyan_correlation_dimension."""
import numpy as np
import pytest
from moirais.fn.rgcrl import rangayyan_correlation_dimension


def test_rgcrl_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = rangayyan_correlation_dimension(x, y)
    assert abs(result['statistic'] - 1.0) < 0.01


def test_rgcrl_edge():
    """Test edge cases."""
    result = rangayyan_correlation_dimension(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result['n'] == 2
