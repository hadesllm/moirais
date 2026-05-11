"""Tests for gb733.gibbons_linrank_covariance."""
import numpy as np
import pytest
from morie.fn.gb733 import gibbons_linrank_covariance


def test_gb733_basic():
    """Test basic functionality."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    N = 100
    result = gibbons_linrank_covariance(a, b, m, n, N)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb733_edge():
    """Test edge cases."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    N = 100
    result = gibbons_linrank_covariance(a, b, m, n, N)
    assert isinstance(result, dict)
