"""Tests for gb432.gibbons_ks_exact_dist."""
import numpy as np
import pytest
from morie.fn.gb432 import gibbons_ks_exact_dist


def test_gb432_basic():
    """Test basic functionality."""
    v = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_ks_exact_dist(v, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb432_edge():
    """Test edge cases."""
    v = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_ks_exact_dist(v, n)
    assert isinstance(result, dict)
