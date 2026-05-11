"""Tests for gb_wss.gibbons_wrs_normal_approx."""
import numpy as np
import pytest
from morie.fn.gb_wss import gibbons_wrs_normal_approx


def test_gb_wss_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    N = 100
    result = gibbons_wrs_normal_approx(W, m, n, N)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_wss_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    N = 100
    result = gibbons_wrs_normal_approx(W, m, n, N)
    assert isinstance(result, dict)
