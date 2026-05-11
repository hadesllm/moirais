"""Tests for fzecdf.fauzi_ecdf."""
import numpy as np
import pytest
from morie.fn.fzecdf import fauzi_ecdf


def test_fzecdf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_ecdf(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzecdf_edge():
    """Test edge cases."""
    result = fauzi_ecdf(np.array([42.0]))
    assert result['n'] == 1
