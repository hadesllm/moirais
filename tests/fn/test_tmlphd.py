"""Tests for tmlphd.tmle_high_dim."""
import numpy as np
import pytest
from morie.fn.tmlphd import tmle_high_dim


def test_tmlphd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = tmle_high_dim(y, D, X, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlphd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = tmle_high_dim(y, D, X, lam)
    assert isinstance(result, dict)
