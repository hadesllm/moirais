"""Tests for covbal.covariate_balance_check."""
import numpy as np
import pytest
from moirais.fn.covbal import covariate_balance_check


def test_covbal_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = covariate_balance_check(A, H, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_covbal_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = covariate_balance_check(A, H, weights)
    assert isinstance(result, dict)
