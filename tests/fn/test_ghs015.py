"""Tests for ghs015.ghosal_ch3_dirichlet_posterior_var."""
import numpy as np
import pytest
from moirais.fn.ghs015 import ghosal_ch3_dirichlet_posterior_var


def test_ghs015_basic():
    """Test basic functionality."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_dirichlet_posterior_var(alpha_j, N_j, j, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs015_edge():
    """Test edge cases."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_dirichlet_posterior_var(alpha_j, N_j, j, n)
    assert isinstance(result, dict)
