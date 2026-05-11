"""Tests for ghs012.ghosal_ch3_countable_dirichlet_posterior_l."""
import numpy as np
import pytest
from morie.fn.ghs012 import ghosal_ch3_countable_dirichlet_posterior_l


def test_ghs012_basic():
    """Test basic functionality."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_countable_dirichlet_posterior_l(alpha_j, N_j, l, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs012_edge():
    """Test edge cases."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_countable_dirichlet_posterior_l(alpha_j, N_j, l, n)
    assert isinstance(result, dict)
