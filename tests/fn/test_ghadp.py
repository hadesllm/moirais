"""Tests for ghadp.ghosal_adaptation."""
import numpy as np
import pytest
from moirais.fn.ghadp import ghosal_adaptation


def test_ghadp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_adaptation(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghadp_edge():
    """Test edge cases."""
    result = ghosal_adaptation(np.array([42.0]))
    assert result['n'] == 1
