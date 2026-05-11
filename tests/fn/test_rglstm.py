"""Tests for rglstm.rangayyan_lstm_signal."""
import numpy as np
import pytest
from morie.fn.rglstm import rangayyan_lstm_signal


def test_rglstm_basic():
    """Test basic functionality."""
    X_seq = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    hidden_size = 100
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_lstm_signal(X_seq, y, hidden_size, n_layers, lr, epochs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rglstm_edge():
    """Test edge cases."""
    X_seq = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    hidden_size = 100
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_lstm_signal(X_seq, y, hidden_size, n_layers, lr, epochs)
    assert isinstance(result, dict)
