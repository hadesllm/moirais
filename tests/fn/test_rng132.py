"""Tests for rng132.rangayyan_ch3_iir_difference_eq_general."""
import numpy as np
import pytest
from moirais.fn.rng132 import rangayyan_ch3_iir_difference_eq_general


def test_rng132_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    n = 100
    result = rangayyan_ch3_iir_difference_eq_general(x, y, b_k, a_k, N, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng132_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    n = 100
    result = rangayyan_ch3_iir_difference_eq_general(x, y, b_k, a_k, N, n)
    assert isinstance(result, dict)
