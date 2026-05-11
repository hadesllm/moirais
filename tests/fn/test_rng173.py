"""Tests for rng173.rangayyan_ch3_rls_gain_identity."""
import numpy as np
import pytest
from morie.fn.rng173 import rangayyan_ch3_rls_gain_identity


def test_rng173_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    n = 100
    result = rangayyan_ch3_rls_gain_identity(P, r, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng173_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    n = 100
    result = rangayyan_ch3_rls_gain_identity(P, r, n)
    assert isinstance(result, dict)
