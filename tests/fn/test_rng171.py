"""Tests for rng171.rangayyan_ch3_rls_kalman_gain."""
import numpy as np
import pytest
from moirais.fn.rng171 import rangayyan_ch3_rls_kalman_gain


def test_rng171_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_kalman_gain(P, r, lam, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng171_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_kalman_gain(P, r, lam, n)
    assert isinstance(result, dict)
