"""Tests for latnh.latin_hypercube."""
import numpy as np
import pytest
from morie.fn.latnh import latin_hypercube


def test_latnh_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = latin_hypercube(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_latnh_edge():
    """Test edge cases."""
    result = latin_hypercube(np.array([42.0]))
    assert result['n'] == 1
