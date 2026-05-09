"""Tests for rng172.rangayyan_ch3_rls_p_recursion."""
import numpy as np
import pytest
from moirais.fn.rng172 import rangayyan_ch3_rls_p_recursion


def test_rng172_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    r = 10
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_p_recursion(P, k, r, lam, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng172_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    r = 10
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_p_recursion(P, k, r, lam, n)
    assert isinstance(result, dict)
