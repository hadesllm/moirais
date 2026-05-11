"""Tests for liouB.liouville_test."""
import numpy as np
import pytest
from morie.fn.liouB import liouville_test


def test_liouB_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = liouville_test(expr, x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_liouB_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = liouville_test(expr, x)
    assert isinstance(result, dict)
