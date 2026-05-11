"""Tests for wdemb.word_embedding."""
import numpy as np
import pytest
from morie.fn.wdemb import word_embedding


def test_wdemb_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = word_embedding(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_wdemb_edge():
    """Test edge cases."""
    result = word_embedding(np.array([42.0]))
    assert result['n'] == 1
