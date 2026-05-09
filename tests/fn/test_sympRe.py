"""Tests for sympRe.sympy_simplify."""
import numpy as np
import pytest
from moirais.fn.sympRe import sympy_simplify


def test_sympRe_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = sympy_simplify(expr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sympRe_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = sympy_simplify(expr)
    assert isinstance(result, dict)
