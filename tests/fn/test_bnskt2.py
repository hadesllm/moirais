"""Tests for bnskt2.bound_kink_te."""
import numpy as np
import pytest
from morie.fn.bnskt2 import bound_kink_te


def test_bnskt2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = bound_kink_te(y, x, cutoff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnskt2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = bound_kink_te(y, x, cutoff)
    assert isinstance(result, dict)
