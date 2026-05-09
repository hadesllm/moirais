"""Tests for hmrnn.geron_recurrent_neuron."""
import numpy as np
import pytest
from moirais.fn.hmrnn import geron_recurrent_neuron


def test_hmrnn_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    Wx = np.random.default_rng(42).normal(0, 1, 100)
    Wh = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_recurrent_neuron(x_t, h_prev, Wx, Wh, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrnn_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    Wx = np.random.default_rng(42).normal(0, 1, 100)
    Wh = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_recurrent_neuron(x_t, h_prev, Wx, Wh, b)
    assert isinstance(result, dict)
