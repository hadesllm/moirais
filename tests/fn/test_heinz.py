"""Tests for heinz.he_initialization."""
import numpy as np
import pytest
from morie.fn.heinz import he_initialization


def test_heinz_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = he_initialization(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_heinz_edge():
    """Test edge cases."""
    result = he_initialization(np.array([42.0]))
    assert result['n'] == 1
