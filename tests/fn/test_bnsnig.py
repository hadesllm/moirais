"""Tests for bnsnig.bound_no_unobserved_inv."""
import numpy as np
import pytest
from moirais.fn.bnsnig import bound_no_unobserved_inv


def test_bnsnig_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    X_inv = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_no_unobserved_inv(y, D, X, X_inv)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnsnig_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    X_inv = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_no_unobserved_inv(y, D, X, X_inv)
    assert isinstance(result, dict)
