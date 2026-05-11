"""Tests for gb_spq.gibbons_spearman_test."""
import numpy as np
import pytest
from morie.fn.gb_spq import gibbons_spearman_test


def test_gb_spq_basic():
    """Test basic functionality."""
    r_s = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_spearman_test(r_s, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_spq_edge():
    """Test edge cases."""
    r_s = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_spearman_test(r_s, n)
    assert isinstance(result, dict)
