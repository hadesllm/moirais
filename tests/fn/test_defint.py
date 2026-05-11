"""Tests for defint.definite_integral."""
import numpy as np
import pytest
from morie.fn.defint import definite_integral


def test_defint_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = definite_integral(expr, x, a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_defint_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = definite_integral(expr, x, a, b)
    assert isinstance(result, dict)
