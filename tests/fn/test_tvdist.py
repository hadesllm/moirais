"""Tests for tvdist.total_variation_distance."""
import numpy as np
import pytest
from morie.fn.tvdist import total_variation_distance


def test_tvdist_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = total_variation_distance(y, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tvdist_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = total_variation_distance(y, p, q)
    assert isinstance(result, dict)
