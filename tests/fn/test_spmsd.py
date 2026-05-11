"""Tests for spmsd.schabenberger_ms_differentiability."""
import numpy as np
import pytest
from morie.fn.spmsd import schabenberger_ms_differentiability


def test_spmsd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = schabenberger_ms_differentiability(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_spmsd_edge():
    """Test edge cases."""
    result = schabenberger_ms_differentiability(np.array([42.0]))
    assert result['n'] == 1
