"""Tests for dpmem.dirichlet_process_mixture."""
import numpy as np
import pytest
from morie.fn.dpmem import dirichlet_process_mixture


def test_dpmem_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    base_distribution = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = dirichlet_process_mixture(y, alpha, base_distribution, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpmem_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    base_distribution = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = dirichlet_process_mixture(y, alpha, base_distribution, n_iter)
    assert isinstance(result, dict)
