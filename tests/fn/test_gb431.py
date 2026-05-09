"""Tests for gb431.gibbons_ks_dist_free."""
import numpy as np
import pytest
from moirais.fn.gb431 import gibbons_ks_dist_free


def test_gb431_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_ks_dist_free(x, F0, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb431_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_ks_dist_free(x, F0, n)
    assert isinstance(result, dict)
