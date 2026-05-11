"""Tests for rng175.rangayyan_ch3_rls_a_priori_error."""
import numpy as np
import pytest
from morie.fn.rng175 import rangayyan_ch3_rls_a_priori_error


def test_rng175_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    w_tilde = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_rls_a_priori_error(x, r, w_tilde, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng175_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    w_tilde = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_rls_a_priori_error(x, r, w_tilde, n)
    assert isinstance(result, dict)
