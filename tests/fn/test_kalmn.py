"""Tests for kalmn.kalman_filter."""
import numpy as np
import pytest
from moirais.fn.kalmn import kalman_filter


def test_kalmn_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kalman_filter(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_kalmn_edge():
    """Test edge cases."""
    result = kalman_filter(np.array([42.0]))
    assert result['n'] == 1
