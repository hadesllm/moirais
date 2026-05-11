"""Tests for icfR.item_cf."""
import numpy as np
import pytest
from morie.fn.icfR import item_cf


def test_icfR_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = item_cf(R, u, i, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icfR_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = item_cf(R, u, i, k)
    assert isinstance(result, dict)
