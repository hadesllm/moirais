"""Tests for hmsrnn.geron_simple_rnn."""
import numpy as np
import pytest
from moirais.fn.hmsrnn import geron_simple_rnn


def test_hmsrnn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Wx = np.random.default_rng(42).normal(0, 1, 100)
    Wh = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_simple_rnn(X, Wx, Wh, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsrnn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Wx = np.random.default_rng(42).normal(0, 1, 100)
    Wh = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_simple_rnn(X, Wx, Wh, b)
    assert isinstance(result, dict)
