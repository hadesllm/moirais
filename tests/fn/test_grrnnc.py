"""Tests for grrnnc.geron_simple_rnn_cell."""
import numpy as np
import pytest
from moirais.fn.grrnnc import geron_simple_rnn_cell


def test_grrnnc_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    Whh = np.random.default_rng(42).normal(0, 1, 100)
    Wxh = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_simple_rnn_cell(x_t, h_prev, Whh, Wxh, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grrnnc_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    Whh = np.random.default_rng(42).normal(0, 1, 100)
    Wxh = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_simple_rnn_cell(x_t, h_prev, Whh, Wxh, b)
    assert isinstance(result, dict)
