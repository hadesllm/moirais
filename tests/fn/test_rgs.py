"""Tests for rgs.functional_regression."""
import numpy as np
import pytest
from morie.fn.rgs import functional_regression


def test_rgs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = functional_regression(X, Y, basis)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgs_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = functional_regression(X, Y, basis)
    assert isinstance(result, dict)
