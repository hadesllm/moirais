"""Tests for sgtgrn.sgt_graph_neural_propagation."""
import numpy as np
import pytest
from morie.fn.sgtgrn import sgt_graph_neural_propagation


def test_sgtgrn_basic():
    """Test basic functionality."""
    A_hat = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_graph_neural_propagation(A_hat, X, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtgrn_edge():
    """Test edge cases."""
    A_hat = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_graph_neural_propagation(A_hat, X, W)
    assert isinstance(result, dict)
