"""Tests for cncrd.concordance_incomplete."""
import numpy as np
import pytest
from morie.fn.cncrd import concordance_incomplete


def test_cncrd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = concordance_incomplete(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_cncrd_edge():
    """Test edge cases."""
    result = concordance_incomplete(np.array([42.0]))
    assert result['n'] == 1
