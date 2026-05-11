"""Tests for cgmth.conjugate_gradient."""
import numpy as np
import pytest
from morie.fn.cgmth import cgmth as conjugate_gradient


def test_cgmth_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = conjugate_gradient(A, b, x0, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cgmth_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = conjugate_gradient(A, b, x0, tol)
    assert isinstance(result, dict)
