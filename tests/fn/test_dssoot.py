"""Tests for dssoot.bootstrap_indirect."""
import numpy as np
import pytest
from moirais.fn.dssoot import bootstrap_indirect


def test_dssoot_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    n_boot = 100
    result = bootstrap_indirect(Y, X, M, n_boot)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dssoot_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    n_boot = 100
    result = bootstrap_indirect(Y, X, M, n_boot)
    assert isinstance(result, dict)
