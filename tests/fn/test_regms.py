"""Tests for regms.regime_switching."""
import numpy as np
import pytest
from morie.fn.regms import regime_switching


def test_regms_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = regime_switching(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_regms_edge():
    """Test edge cases."""
    result = regime_switching(np.array([42.0]))
    assert result['n'] == 1
