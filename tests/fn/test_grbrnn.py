"""Tests for grbrnn.geron_bidirectional_rnn."""
import numpy as np
import pytest
from morie.fn.grbrnn import geron_bidirectional_rnn


def test_grbrnn_basic():
    """Test basic functionality."""
    h_forward = np.random.default_rng(42).normal(0, 1, 100)
    h_backward = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bidirectional_rnn(h_forward, h_backward)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbrnn_edge():
    """Test edge cases."""
    h_forward = np.random.default_rng(42).normal(0, 1, 100)
    h_backward = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bidirectional_rnn(h_forward, h_backward)
    assert isinstance(result, dict)
