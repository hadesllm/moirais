"""Tests for spcont.schabenberger_spatial_continuity."""
import numpy as np
import pytest
from morie.fn.spcont import schabenberger_spatial_continuity


def test_spcont_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = schabenberger_spatial_continuity(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_spcont_edge():
    """Test edge cases."""
    result = schabenberger_spatial_continuity(np.array([42.0]))
    assert result['n'] == 1
