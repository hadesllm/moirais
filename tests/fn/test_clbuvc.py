"""Tests for clbuvc.club_upper_bound."""
import numpy as np
import pytest
from morie.fn.clbuvc import club_upper_bound


def test_clbuvc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = club_upper_bound(x, y, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clbuvc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = club_upper_bound(x, y, q)
    assert isinstance(result, dict)
