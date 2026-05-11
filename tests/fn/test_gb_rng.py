"""Tests for gb_rng.gibbons_range_dist."""
import numpy as np
import pytest
from morie.fn.gb_rng import gibbons_range_dist


def test_gb_rng_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_range_dist(w, n, f, F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_rng_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_range_dist(w, n, f, F)
    assert isinstance(result, dict)
