"""Tests for bkelm.burkov_elman_rnn."""
import numpy as np
import pytest
from moirais.fn.bkelm import burkov_elman_rnn


def test_bkelm_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    Wh = np.random.default_rng(42).normal(0, 1, 100)
    Wx = np.random.default_rng(42).normal(0, 1, 100)
    Wy = np.random.default_rng(42).normal(0, 1, 100)
    bh = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_elman_rnn(x_t, h_prev, Wh, Wx, Wy, bh, by)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkelm_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    Wh = np.random.default_rng(42).normal(0, 1, 100)
    Wx = np.random.default_rng(42).normal(0, 1, 100)
    Wy = np.random.default_rng(42).normal(0, 1, 100)
    bh = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_elman_rnn(x_t, h_prev, Wh, Wx, Wy, bh, by)
    assert isinstance(result, dict)
