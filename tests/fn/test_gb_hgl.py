"""Tests for gb_hgl.gibbons_hodges_lehmann."""
import numpy as np
import pytest
from moirais.fn.gb_hgl import gibbons_hodges_lehmann


def test_gb_hgl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_hodges_lehmann(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_hgl_edge():
    """Test edge cases."""
    result = gibbons_hodges_lehmann(np.array([42.0]))
    assert result['n'] == 1
