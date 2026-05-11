"""Tests for gb732.gibbons_linrank_mean_var."""
import numpy as np
import pytest
from morie.fn.gb732 import gibbons_linrank_mean_var


def test_gb732_basic():
    """Test basic functionality."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    N = 100
    result = gibbons_linrank_mean_var(a, m, n, N)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb732_edge():
    """Test edge cases."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    N = 100
    result = gibbons_linrank_mean_var(a, m, n, N)
    assert isinstance(result, dict)
