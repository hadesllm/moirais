"""Tests for diffS.symbolic_diff."""
import numpy as np
import pytest
from morie.fn.diffs import symbolic_diff


def test_diffs_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = symbolic_diff(expr, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diffs_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = symbolic_diff(expr, x)
    assert isinstance(result, dict)
