"""Tests for dpgmm.dp_gaussian_mixture."""
import numpy as np
import pytest
from morie.fn.dpgmm import dp_gaussian_mixture


def test_dpgmm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    prior_mu = np.random.default_rng(42).normal(0, 1, 100)
    prior_sigma = np.random.default_rng(42).normal(0, 1, 100)
    truncation = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_gaussian_mixture(y, alpha, prior_mu, prior_sigma, truncation)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpgmm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    prior_mu = np.random.default_rng(42).normal(0, 1, 100)
    prior_sigma = np.random.default_rng(42).normal(0, 1, 100)
    truncation = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_gaussian_mixture(y, alpha, prior_mu, prior_sigma, truncation)
    assert isinstance(result, dict)
