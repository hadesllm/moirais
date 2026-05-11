"""Tests for agphdt.alphazero_policy_head."""
import numpy as np
import pytest
from morie.fn.agphdt import alphazero_policy_head


def test_agphdt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    action_space = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_policy_head(x, action_space)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agphdt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    action_space = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_policy_head(x, action_space)
    assert isinstance(result, dict)
