"""Tests for alpz.alphazero_search."""
import numpy as np
import pytest
from moirais.fn.alpz import alphazero_search


def test_alpz_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    net = np.random.default_rng(42).normal(0, 1, 100)
    num_sim = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_search(state, net, num_sim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alpz_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    net = np.random.default_rng(42).normal(0, 1, 100)
    num_sim = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_search(state, net, num_sim)
    assert isinstance(result, dict)
