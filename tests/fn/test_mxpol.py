"""Tests for mxpol.maxpool_forward."""
import numpy as np
import pytest
from moirais.fn.mxpol import maxpool_forward


def test_mxpol_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = maxpool_forward(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mxpol_edge():
    """Test edge cases."""
    result = maxpool_forward(np.array([42.0]))
    assert result['n'] == 1
