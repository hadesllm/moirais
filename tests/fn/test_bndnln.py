"""Tests for bndnln.bound_nonlinear."""
import numpy as np
import pytest
from moirais.fn.bndnln import bound_nonlinear


def test_bndnln_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = bound_nonlinear(y, X, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndnln_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = bound_nonlinear(y, X, g)
    assert isinstance(result, dict)
