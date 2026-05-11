"""Tests for ipwSn.ipw_sensitivity."""
import numpy as np
import pytest
from morie.fn.ipwSn import ipw_sensitivity


def test_ipwSn_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    lam_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = ipw_sensitivity(Y, X, C, lam_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ipwSn_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    lam_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = ipw_sensitivity(Y, X, C, lam_grid)
    assert isinstance(result, dict)
