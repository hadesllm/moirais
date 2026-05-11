"""Tests for limT.symbolic_limit."""
import numpy as np
import pytest
from morie.fn.limT import symbolic_limit


def test_limT_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = symbolic_limit(expr, x, x0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_limT_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = symbolic_limit(expr, x, x0)
    assert isinstance(result, dict)
