"""Tests for grpqa.grouped_query_attention."""
import numpy as np
import pytest
from moirais.fn.grpqa import grouped_query_attention


def test_grpqa_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = grouped_query_attention(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_grpqa_edge():
    """Test edge cases."""
    result = grouped_query_attention(np.array([42.0]))
    assert result['n'] == 1
