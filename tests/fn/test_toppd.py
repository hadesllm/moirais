"""Tests for toppd.top_p_nucleus."""
import numpy as np
import pytest
from moirais.fn.toppd import top_p_nucleus


def test_toppd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = top_p_nucleus(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_toppd_edge():
    """Test edge cases."""
    result = top_p_nucleus(np.array([42.0]))
    assert result['n'] == 1
