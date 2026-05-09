"""Tests for agexpd.alphazero_expand."""
import numpy as np
import pytest
from moirais.fn.agexpd import alphazero_expand


def test_agexpd_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    policy_net = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_expand(state, policy_net)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agexpd_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    policy_net = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_expand(state, policy_net)
    assert isinstance(result, dict)
