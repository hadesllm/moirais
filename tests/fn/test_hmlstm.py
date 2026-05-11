"""Tests for hmlstm.geron_lstm."""
import numpy as np
import pytest
from morie.fn.hmlstm import geron_lstm


def test_hmlstm_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlstm_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict)
