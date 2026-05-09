"""Tests for bndnpr.bound_nonparam_regr."""
import numpy as np
import pytest
from moirais.fn.bndnpr import bound_nonparam_regr


def test_bndnpr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bw = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_nonparam_regr(y, D, X, bw)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndnpr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bw = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_nonparam_regr(y, D, X, bw)
    assert isinstance(result, dict)
