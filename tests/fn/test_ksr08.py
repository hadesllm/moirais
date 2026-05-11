"""Tests for ksr08.kosorok_multiplier_bootstrap."""
import numpy as np
import pytest
from morie.fn.ksr08 import kosorok_multiplier_bootstrap


def test_ksr08_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_multiplier_bootstrap(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ksr08_edge():
    """Test edge cases."""
    result = kosorok_multiplier_bootstrap(np.array([42.0]))
    assert result['n'] == 1
