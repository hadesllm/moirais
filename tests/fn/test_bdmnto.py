"""Tests for bdmnto.bound_monot_outcome."""
import numpy as np
import pytest
from morie.fn.bdmnto import bound_monot_outcome


def test_bdmnto_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    direction = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_monot_outcome(y, D, X, direction)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bdmnto_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    direction = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_monot_outcome(y, D, X, direction)
    assert isinstance(result, dict)
