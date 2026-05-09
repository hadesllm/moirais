"""Tests for bndnmt.bound_no_monotonicity."""
import numpy as np
import pytest
from moirais.fn.bndnmt import bound_no_monotonicity


def test_bndnmt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = bound_no_monotonicity(y, D, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndnmt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = bound_no_monotonicity(y, D, Z)
    assert isinstance(result, dict)
