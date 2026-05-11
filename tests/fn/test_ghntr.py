"""Tests for ghntr.ghosal_neutral_right."""
import numpy as np
import pytest
from morie.fn.ghntr import ghosal_neutral_right


def test_ghntr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_neutral_right(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghntr_edge():
    """Test edge cases."""
    result = ghosal_neutral_right(np.array([42.0]))
    assert result['n'] == 1
