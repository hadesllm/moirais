"""Tests for rslnk.residual_connection."""
import numpy as np
import pytest
from morie.fn.rslnk import residual_connection


def test_rslnk_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = residual_connection(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rslnk_edge():
    """Test edge cases."""
    result = residual_connection(np.array([42.0]))
    assert result['n'] == 1
