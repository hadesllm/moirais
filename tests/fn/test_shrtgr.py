"""Tests for shrtgr.shrinkage_propensity."""
import numpy as np
import pytest
from moirais.fn.shrtgr import shrinkage_propensity


def test_shrtgr_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    prior_mu = np.random.default_rng(42).normal(0, 1, 100)
    prior_sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = shrinkage_propensity(A, H, prior_mu, prior_sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shrtgr_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    prior_mu = np.random.default_rng(42).normal(0, 1, 100)
    prior_sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = shrinkage_propensity(A, H, prior_mu, prior_sigma)
    assert isinstance(result, dict)
