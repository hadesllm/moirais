"""Tests for posab.positional_encoding_abs."""
import numpy as np
import pytest
from morie.fn.posab import positional_encoding_abs


def test_posab_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = positional_encoding_abs(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_posab_edge():
    """Test edge cases."""
    result = positional_encoding_abs(np.array([42.0]))
    assert result['n'] == 1
