"""A journey of a thousand miles begins with a single step. — Lao Tzu"""
import numpy as np
import pytest
from morie.fn.flshA import flash_attention


def test_flshA_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = flash_attention(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_flshA_edge():
    """Test edge cases."""
    result = flash_attention(np.array([42.0]))
    assert result['n'] == 1
