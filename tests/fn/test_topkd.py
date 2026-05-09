"""Tests for topkd.top_k_decoding."""
import numpy as np
import pytest
from moirais.fn.topkd import top_k_decoding


def test_topkd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = top_k_decoding(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_topkd_edge():
    """Test edge cases."""
    result = top_k_decoding(np.array([42.0]))
    assert result['n'] == 1
