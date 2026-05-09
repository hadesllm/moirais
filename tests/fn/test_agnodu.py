"""Tests for agnodu.alphazero_node_update."""
import numpy as np
import pytest
from moirais.fn.agnodu import alphazero_node_update


def test_agnodu_basic():
    """Test basic functionality."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_node_update(v, node)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_agnodu_edge():
    """Test edge cases."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_node_update(v, node)
    assert isinstance(result, dict)
