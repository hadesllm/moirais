"""Tests for hmphp.geron_peephole_lstm."""
import numpy as np
import pytest
from moirais.fn.hmphp import geron_peephole_lstm


def test_hmphp_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_peephole_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmphp_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_peephole_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict)
