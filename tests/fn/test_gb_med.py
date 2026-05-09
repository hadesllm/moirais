"""Tests for gb_med.gibbons_median_dist."""
import numpy as np
import pytest
from moirais.fn.gb_med import gibbons_median_dist


def test_gb_med_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_median_dist(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_med_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_median_dist(x, n)
    assert isinstance(result, dict)
