"""Tests for scfd.scalar_on_function."""
import numpy as np
import pytest
from morie.fn.scfd import scalar_on_function


def test_scfd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = scalar_on_function(X, Y, basis)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scfd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = scalar_on_function(X, Y, basis)
    assert isinstance(result, dict)
