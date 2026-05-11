"""Tests for grucl.gru_cell."""
import numpy as np
import pytest
from morie.fn.grucl import gru_cell


def test_grucl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gru_cell(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_grucl_edge():
    """Test edge cases."""
    result = gru_cell(np.array([42.0]))
    assert result['n'] == 1
