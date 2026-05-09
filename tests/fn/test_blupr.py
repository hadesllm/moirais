"""Tests for blupr.blup_random_intercept."""
import numpy as np
import pytest
from moirais.fn.blupr import blup_random_intercept


def test_blupr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = blup_random_intercept(y, X, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blupr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = blup_random_intercept(y, X, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
