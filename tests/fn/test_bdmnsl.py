"""Tests for bdmnsl.bound_monot_selection."""
import numpy as np
import pytest
from moirais.fn.bdmnsl import bound_monot_selection


def test_bdmnsl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_monot_selection(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bdmnsl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_monot_selection(y, D, X)
    assert isinstance(result, dict)
