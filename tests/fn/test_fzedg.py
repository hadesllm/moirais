"""Tests for fzedg.fauzi_edgeworth_quantile."""
import numpy as np
import pytest
from morie.fn.fzedg import fauzi_edgeworth_quantile


def test_fzedg_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_edgeworth_quantile(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzedg_edge():
    """Test edge cases."""
    result = fauzi_edgeworth_quantile(np.array([42.0]))
    assert result['n'] == 1
