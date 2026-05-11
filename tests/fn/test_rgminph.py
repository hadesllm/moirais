"""Tests for rgminph.rangayyan_min_phase."""
import numpy as np
import pytest
from morie.fn.rgminph import rangayyan_min_phase


def test_rgminph_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_min_phase(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgminph_edge():
    """Test edge cases."""
    result = rangayyan_min_phase(np.array([42.0]))
    assert result['n'] == 1
