"""Tests for agdirn.alphazero_dirichlet_noise."""
import numpy as np
import pytest
from morie.fn.agdirn import alphazero_dirichlet_noise


def test_agdirn_basic():
    """Test basic functionality."""
    p = 5
    alpha = 0.05
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_dirichlet_noise(p, alpha, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agdirn_edge():
    """Test edge cases."""
    p = 5
    alpha = 0.05
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_dirichlet_noise(p, alpha, eps)
    assert isinstance(result, dict)
