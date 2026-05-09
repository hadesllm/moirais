"""Tests for irtgpc.generalized_partial_credit."""
import numpy as np
import pytest
from moirais.fn.irtgpc import generalized_partial_credit


def test_irtgpc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = generalized_partial_credit(X, ncats)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irtgpc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = generalized_partial_credit(X, ncats)
    assert isinstance(result, dict)
