"""Tests for mhatf.multi_head_attention_full."""
import numpy as np
import pytest
from moirais.fn.mhatf import multi_head_attention_full


def test_mhatf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = multi_head_attention_full(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mhatf_edge():
    """Test edge cases."""
    result = multi_head_attention_full(np.array([42.0]))
    assert result['n'] == 1
