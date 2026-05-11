"""Tests for ksr04.kosorok_vc_dimension."""
import numpy as np
import pytest
from morie.fn.ksr04 import kosorok_vc_dimension


def test_ksr04_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_vc_dimension(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ksr04_edge():
    """Test edge cases."""
    result = kosorok_vc_dimension(np.array([42.0]))
    assert result['n'] == 1
