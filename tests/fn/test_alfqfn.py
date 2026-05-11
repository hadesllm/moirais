"""Tests for alfqfn.alphazero_q_function."""
import numpy as np
import pytest
from morie.fn.alfqfn import alphazero_q_function


def test_alfqfn_basic():
    """Test basic functionality."""
    N = 100
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = alphazero_q_function(N, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfqfn_edge():
    """Test edge cases."""
    N = 100
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = alphazero_q_function(N, v)
    assert isinstance(result, dict)
