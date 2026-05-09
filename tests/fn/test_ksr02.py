"""Tests for ksr02.kosorok_donsker_class."""
import numpy as np
import pytest
from moirais.fn.ksr02 import kosorok_donsker_class


def test_ksr02_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_donsker_class(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ksr02_edge():
    """Test edge cases."""
    result = kosorok_donsker_class(np.array([42.0]))
    assert result['n'] == 1
