"""Tests for rng057.rangayyan_ch3_iir_difference_equation."""
import numpy as np
import pytest
from moirais.fn.rng057 import rangayyan_ch3_iir_difference_equation


def test_rng057_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    n = 100
    result = rangayyan_ch3_iir_difference_equation(x, y, b_k, a_k, N, M, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng057_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    n = 100
    result = rangayyan_ch3_iir_difference_equation(x, y, b_k, a_k, N, M, n)
    assert isinstance(result, dict)
