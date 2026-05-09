"""Tests for hmdrnn.geron_deep_rnn."""
import numpy as np
import pytest
from moirais.fn.hmdrnn import geron_deep_rnn


def test_hmdrnn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    hidden_sizes = np.random.default_rng(42).normal(0, 1, 100)
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_deep_rnn(X, hidden_sizes, n_layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdrnn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    hidden_sizes = np.random.default_rng(42).normal(0, 1, 100)
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_deep_rnn(X, hidden_sizes, n_layers)
    assert isinstance(result, dict)
