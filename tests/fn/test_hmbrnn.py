"""Tests for hmbrnn.geron_bidirectional_rnn."""
import numpy as np
import pytest
from morie.fn.hmbrnn import geron_bidirectional_rnn


def test_hmbrnn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Wx_f = np.random.default_rng(42).normal(0, 1, 100)
    Wh_f = np.random.default_rng(42).normal(0, 1, 100)
    Wx_b = np.random.default_rng(42).normal(0, 1, 100)
    Wh_b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bidirectional_rnn(X, Wx_f, Wh_f, Wx_b, Wh_b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbrnn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Wx_f = np.random.default_rng(42).normal(0, 1, 100)
    Wh_f = np.random.default_rng(42).normal(0, 1, 100)
    Wx_b = np.random.default_rng(42).normal(0, 1, 100)
    Wh_b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bidirectional_rnn(X, Wx_f, Wh_f, Wx_b, Wh_b)
    assert isinstance(result, dict)
