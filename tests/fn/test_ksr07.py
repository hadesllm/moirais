"""Tests for ksr07.kosorok_bootstrap_empirical."""
import numpy as np
import pytest
from moirais.fn.ksr07 import kosorok_bootstrap_empirical


def test_ksr07_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_bootstrap_empirical(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ksr07_edge():
    """Test edge cases."""
    result = kosorok_bootstrap_empirical(np.array([42.0]))
    assert result['n'] == 1
