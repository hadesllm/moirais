"""Tests for agdrcn.alphazero_dirichlet_concentration."""
import numpy as np
import pytest
from morie.fn.agdrcn import alphazero_dirichlet_concentration


def test_agdrcn_basic():
    """Test basic functionality."""
    avg_legal = np.random.default_rng(42).normal(0, 1, 100)
    scale = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_dirichlet_concentration(avg_legal, scale)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agdrcn_edge():
    """Test edge cases."""
    avg_legal = np.random.default_rng(42).normal(0, 1, 100)
    scale = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_dirichlet_concentration(avg_legal, scale)
    assert isinstance(result, dict)
