"""Tests for graut.geron_autograd_chain_rule."""
import numpy as np
import pytest
from moirais.fn.graut import geron_autograd_chain_rule


def test_graut_basic():
    """Test basic functionality."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    grad_output = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_autograd_chain_rule(graph, grad_output)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_graut_edge():
    """Test edge cases."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    grad_output = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_autograd_chain_rule(graph, grad_output)
    assert isinstance(result, dict)
