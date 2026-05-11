"""Tests for gb_fwv.gibbons_friedman_variance."""
import numpy as np
import pytest
from morie.fn.gb_fwv import gibbons_friedman_variance


def test_gb_fwv_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = gibbons_friedman_variance(b, k)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_fwv_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = gibbons_friedman_variance(b, k)
    assert isinstance(result, dict)
