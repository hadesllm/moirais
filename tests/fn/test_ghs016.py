"""Tests for ghs016.ghosal_ch3_dirichlet_posterior_cov."""
import numpy as np
import pytest
from morie.fn.ghs016 import ghosal_ch3_dirichlet_posterior_cov


def test_ghs016_basic():
    """Test basic functionality."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    alpha_jprime = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    N_jprime = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_dirichlet_posterior_cov(alpha_j, alpha_jprime, N_j, N_jprime, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs016_edge():
    """Test edge cases."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    alpha_jprime = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    N_jprime = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_dirichlet_posterior_cov(alpha_j, alpha_jprime, N_j, N_jprime, n)
    assert isinstance(result, dict)
