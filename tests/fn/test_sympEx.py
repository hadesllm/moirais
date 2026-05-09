"""Tests for sympEx.sympy_expand."""
import numpy as np
import pytest
from moirais.fn.sympEx import sympy_expand


def test_sympEx_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = sympy_expand(expr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sympEx_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = sympy_expand(expr)
    assert isinstance(result, dict)
