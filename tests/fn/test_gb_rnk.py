"""Tests for gb_rnk.gibbons_rank_def."""
import numpy as np
import pytest
from morie.fn.gb_rnk import gibbons_rank_def


def test_gb_rnk_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_rank_def(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_rnk_edge():
    """Test edge cases."""
    result = gibbons_rank_def(np.array([42.0]))
    assert result['n'] == 1
