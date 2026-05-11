"""Tests for bdtrns.bound_transport."""
import numpy as np
import pytest
from morie.fn.bdtrns import bound_transport


def test_bdtrns_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_transport(y, D, X, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bdtrns_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_transport(y, D, X, S)
    assert isinstance(result, dict)
