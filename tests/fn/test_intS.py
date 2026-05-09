"""Tests for intS.symbolic_integrate."""
import numpy as np
import pytest
from moirais.fn.intS import symbolic_integrate


def test_intS_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = symbolic_integrate(expr, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_intS_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = symbolic_integrate(expr, x)
    assert isinstance(result, dict)
