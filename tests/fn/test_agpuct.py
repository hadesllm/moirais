"""Tests for agpuct.alphazero_puct."""
import numpy as np
import pytest
from moirais.fn.agpuct import alphazero_puct


def test_agpuct_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    Q = np.random.default_rng(42).normal(0, 1, 100)
    c_puct = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_puct(P, N, Q, c_puct)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agpuct_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    Q = np.random.default_rng(42).normal(0, 1, 100)
    c_puct = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_puct(P, N, Q, c_puct)
    assert isinstance(result, dict)
