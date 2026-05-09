"""Tests for ksr03.kosorok_glivenko_cantelli."""
import numpy as np
import pytest
from moirais.fn.ksr03 import kosorok_glivenko_cantelli


def test_ksr03_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_glivenko_cantelli(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ksr03_edge():
    """Test edge cases."""
    result = kosorok_glivenko_cantelli(np.array([42.0]))
    assert result['n'] == 1
