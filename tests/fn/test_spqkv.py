"""Tests for spqkv.sparse_attention."""
import numpy as np
import pytest
from morie.fn.spqkv import sparse_attention


def test_spqkv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sparse_attention(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_spqkv_edge():
    """Test edge cases."""
    result = sparse_attention(np.array([42.0]))
    assert result['n'] == 1
