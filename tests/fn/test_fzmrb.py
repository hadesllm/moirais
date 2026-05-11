"""Tests for fzmrb.fauzi_mrl_boundary_free."""
import numpy as np
import pytest
from morie.fn.fzmrb import fauzi_mrl_boundary_free


def test_fzmrb_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_mrl_boundary_free(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzmrb_edge():
    """Test edge cases."""
    result = fauzi_mrl_boundary_free(np.array([42.0]))
    assert result['n'] == 1
