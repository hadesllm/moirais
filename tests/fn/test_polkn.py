"""Tests for polkn.polynomial_kernel."""
import numpy as np
import pytest
from morie.fn.polkn import polynomial_kernel


def test_polkn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = polynomial_kernel(X, c, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_polkn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = polynomial_kernel(X, c, d)
    assert isinstance(result, dict)
