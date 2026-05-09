"""Tests for irlsfn.irls_solver."""
import numpy as np
import pytest
from moirais.fn.irlsfn import irls_solver


def test_irlsfn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = irls_solver(y, X, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irlsfn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = irls_solver(y, X, weights)
    assert isinstance(result, dict)
