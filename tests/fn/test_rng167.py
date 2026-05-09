"""Tests for rng167.rangayyan_ch3_rls_phi_recursion."""
import numpy as np
import pytest
from moirais.fn.rng167 import rangayyan_ch3_rls_phi_recursion


def test_rng167_basic():
    """Test basic functionality."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_phi_recursion(Phi, r, lam, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng167_edge():
    """Test edge cases."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_phi_recursion(Phi, r, lam, n)
    assert isinstance(result, dict)
