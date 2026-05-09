"""Tests for gb1041t.gibbons_kw_ties."""
import numpy as np
import pytest
from moirais.fn.gb1041t import gibbons_kw_ties


def test_gb1041t_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_kw_ties(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb1041t_edge():
    """Test edge cases."""
    result = gibbons_kw_ties(np.array([42.0]))
    assert result['n'] == 1
