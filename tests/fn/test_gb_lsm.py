"""Tests for gb_lsm.gibbons_large_sample_moments."""
import numpy as np
import pytest
from morie.fn.gb_lsm import gibbons_large_sample_moments


def test_gb_lsm_basic():
    """Test basic functionality."""
    r = 10
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_large_sample_moments(r, n, f, F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_lsm_edge():
    """Test edge cases."""
    r = 10
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_large_sample_moments(r, n, f, F)
    assert isinstance(result, dict)
