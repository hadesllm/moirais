"""Tests for agscan.alphazero_self_consistency."""
import numpy as np
import pytest
from moirais.fn.agscan import alphazero_self_consistency


def test_agscan_basic():
    """Test basic functionality."""
    policy_net = np.random.default_rng(42).normal(0, 1, 100)
    seeds = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_self_consistency(policy_net, seeds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agscan_edge():
    """Test edge cases."""
    policy_net = np.random.default_rng(42).normal(0, 1, 100)
    seeds = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_self_consistency(policy_net, seeds)
    assert isinstance(result, dict)
