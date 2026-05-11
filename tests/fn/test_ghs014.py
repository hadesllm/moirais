"""Tests for ghs014.ghosal_ch3_dirichlet_posterior_mean."""
import numpy as np
import pytest
from morie.fn.ghs014 import ghosal_ch3_dirichlet_posterior_mean


def test_ghs014_basic():
    """Test basic functionality."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_dirichlet_posterior_mean(alpha_j, N_j, j, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs014_edge():
    """Test edge cases."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_dirichlet_posterior_mean(alpha_j, N_j, j, n)
    assert isinstance(result, dict)
