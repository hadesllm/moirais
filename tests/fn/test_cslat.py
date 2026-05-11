"""Tests for cslat.causal_attention_mask."""
import numpy as np
import pytest
from morie.fn.cslat import causal_attention_mask


def test_cslat_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = causal_attention_mask(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_cslat_edge():
    """Test edge cases."""
    result = causal_attention_mask(np.array([42.0]))
    assert result['n'] == 1
