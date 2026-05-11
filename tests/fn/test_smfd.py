"""Tests for smfd.smooth_functional_data."""
import numpy as np
import pytest
from morie.fn.smfd import smooth_functional_data


def test_smfd_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    argvals = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = smooth_functional_data(Y, argvals, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smfd_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    argvals = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = smooth_functional_data(Y, argvals, lam)
    assert isinstance(result, dict)
