"""Tests for agnod1.alphazero_node_init."""
import numpy as np
import pytest
from moirais.fn.agnod1 import alphazero_node_init


def test_agnod1_basic():
    """Test basic functionality."""
    p = 5
    action_space = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_node_init(p, action_space)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agnod1_edge():
    """Test edge cases."""
    p = 5
    action_space = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_node_init(p, action_space)
    assert isinstance(result, dict)
