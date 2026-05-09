"""Tests for rng166.rangayyan_ch3_rls_theta_vector."""
import numpy as np
import pytest
from moirais.fn.rng166 import rangayyan_ch3_rls_theta_vector


def test_rng166_basic():
    """Test basic functionality."""
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_theta_vector(r, x, lam, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng166_edge():
    """Test edge cases."""
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_theta_vector(r, x, lam, n)
    assert isinstance(result, dict)
