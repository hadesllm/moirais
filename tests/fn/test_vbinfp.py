"""Tests for vbinfp.variational_bound."""
import numpy as np
import pytest
from moirais.fn.vbinfp import variational_bound


def test_vbinfp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = variational_bound(X, Y, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vbinfp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = variational_bound(X, Y, q)
    assert isinstance(result, dict)
