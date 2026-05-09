"""Tests for rkhsmt.rkhs_multitrait."""
import numpy as np
import pytest
from moirais.fn.rkhsmt import rkhs_multitrait


def test_rkhsmt_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    n_iter = 50
    result = rkhs_multitrait(Y, K, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rkhsmt_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    n_iter = 50
    result = rkhs_multitrait(Y, K, n_iter)
    assert isinstance(result, dict)
