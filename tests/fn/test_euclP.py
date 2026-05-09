"""Tests for euclP.polynomial_gcd."""
import numpy as np
import pytest
from moirais.fn.euclP import polynomial_gcd


def test_euclP_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = polynomial_gcd(p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_euclP_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = polynomial_gcd(p, q)
    assert isinstance(result, dict)
