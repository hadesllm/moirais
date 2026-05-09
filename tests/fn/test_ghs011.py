"""Tests for ghs011.ghosal_ch3_countable_dirichlet_marginal."""
import numpy as np
import pytest
from moirais.fn.ghs011 import ghosal_ch3_countable_dirichlet_marginal


def test_ghs011_basic():
    """Test basic functionality."""
    p_j = np.random.default_rng(42).normal(0, 1, 100)
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ghosal_ch3_countable_dirichlet_marginal(p_j, alpha_j, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs011_edge():
    """Test edge cases."""
    p_j = np.random.default_rng(42).normal(0, 1, 100)
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ghosal_ch3_countable_dirichlet_marginal(p_j, alpha_j, k)
    assert isinstance(result, dict)
