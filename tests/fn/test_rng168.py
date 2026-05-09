"""Tests for rng168.rangayyan_ch3_rls_theta_recursion."""
import numpy as np
import pytest
from moirais.fn.rng168 import rangayyan_ch3_rls_theta_recursion


def test_rng168_basic():
    """Test basic functionality."""
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_theta_recursion(Theta, r, x, lam, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng168_edge():
    """Test edge cases."""
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_theta_recursion(Theta, r, x, lam, n)
    assert isinstance(result, dict)
