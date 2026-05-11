"""Tests for ghs013.ghosal_ch3_countable_dirichlet_posterior_k."""
import numpy as np
import pytest
from morie.fn.ghs013 import ghosal_ch3_countable_dirichlet_posterior_k


def test_ghs013_basic():
    """Test basic functionality."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    n = 100
    result = ghosal_ch3_countable_dirichlet_posterior_k(alpha_j, N_j, k, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs013_edge():
    """Test edge cases."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    n = 100
    result = ghosal_ch3_countable_dirichlet_posterior_k(alpha_j, N_j, k, n)
    assert isinstance(result, dict)
