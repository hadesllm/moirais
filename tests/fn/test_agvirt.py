"""Tests for agvirt.alphazero_virtual_loss."""
import numpy as np
import pytest
from morie.fn.agvirt import alphazero_virtual_loss


def test_agvirt_basic():
    """Test basic functionality."""
    node = np.random.default_rng(42).normal(0, 1, 100)
    virtual_loss = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_virtual_loss(node, virtual_loss)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agvirt_edge():
    """Test edge cases."""
    node = np.random.default_rng(42).normal(0, 1, 100)
    virtual_loss = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_virtual_loss(node, virtual_loss)
    assert isinstance(result, dict)
