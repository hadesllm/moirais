"""Tests for lstmc.lstm_cell."""
import numpy as np
import pytest
from morie.fn.lstmc import lstm_cell


def test_lstmc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = lstm_cell(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_lstmc_edge():
    """Test edge cases."""
    result = lstm_cell(np.array([42.0]))
    assert result['n'] == 1
