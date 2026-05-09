"""Tests for ksr06.kosorok_maximal_inequality."""
import numpy as np
import pytest
from moirais.fn.ksr06 import kosorok_maximal_inequality


def test_ksr06_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_maximal_inequality(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ksr06_edge():
    """Test edge cases."""
    result = kosorok_maximal_inequality(np.array([42.0]))
    assert result['n'] == 1
