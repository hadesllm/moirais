"""Tests for sympFa.sympy_factor."""
import numpy as np
import pytest
from morie.fn.sympFa import sympy_factor


def test_sympFa_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = sympy_factor(expr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sympFa_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = sympy_factor(expr)
    assert isinstance(result, dict)
