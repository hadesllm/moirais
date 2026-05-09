"""Tests for dwnmn.dynamic_wnominate."""
import numpy as np
import pytest
from moirais.fn.dwnmn import dynamic_wnominate


def test_dwnmn_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = dynamic_wnominate(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_dwnmn_edge():
    """Test edge cases."""
    result = dynamic_wnominate(np.array([42.0]))
    assert result['n'] == 1
