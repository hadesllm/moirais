"""Tests for attnq.scaled_dot_product_attention."""
import numpy as np
import pytest
from moirais.fn.attnq import scaled_dot_product_attention


def test_attnq_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = scaled_dot_product_attention(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_attnq_edge():
    """Test edge cases."""
    result = scaled_dot_product_attention(np.array([42.0]))
    assert result['n'] == 1
