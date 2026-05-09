"""Tests for ksr05.kosorok_bracketing_number."""
import numpy as np
import pytest
from moirais.fn.ksr05 import kosorok_bracketing_number


def test_ksr05_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_bracketing_number(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ksr05_edge():
    """Test edge cases."""
    result = kosorok_bracketing_number(np.array([42.0]))
    assert result['n'] == 1
