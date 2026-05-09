"""Tests for blups.blup_random_slope."""
import numpy as np
import pytest
from moirais.fn.blups import blup_random_slope


def test_blups_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = blup_random_slope(y, X, Z, cluster, D, sigma2_e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blups_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = blup_random_slope(y, X, Z, cluster, D, sigma2_e)
    assert isinstance(result, dict)
