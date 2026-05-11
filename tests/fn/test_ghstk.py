"""Tests for ghstk.ghosal_stick_breaking_trunc."""
import numpy as np
import pytest
from morie.fn.ghstk import ghosal_stick_breaking_trunc


def test_ghstk_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_stick_breaking_trunc(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghstk_edge():
    """Test edge cases."""
    result = ghosal_stick_breaking_trunc(np.array([42.0]))
    assert result['n'] == 1
