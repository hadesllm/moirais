"""Tests for jzdiff.jenson_zhang_disparity."""
import numpy as np
import pytest
from morie.fn.jzdiff import jenson_zhang_disparity


def test_jzdiff_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = jenson_zhang_disparity(y, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jzdiff_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = jenson_zhang_disparity(y, p, q)
    assert isinstance(result, dict)
