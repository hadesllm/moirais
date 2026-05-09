"""Tests for irtsp.irt_spatial."""
import numpy as np
import pytest
from moirais.fn.irtsp import irt_spatial


def test_irtsp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = irt_spatial(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_irtsp_edge():
    """Test edge cases."""
    result = irt_spatial(np.array([42.0]))
    assert result['n'] == 1
