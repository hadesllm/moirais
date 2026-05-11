"""Tests for grdcl.gradient_clipping."""
import numpy as np
import pytest
from morie.fn.grdcl import gradient_clipping


def test_grdcl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gradient_clipping(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_grdcl_edge():
    """Test edge cases."""
    result = gradient_clipping(np.array([42.0]))
    assert result['n'] == 1
