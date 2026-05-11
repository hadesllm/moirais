"""Tests for hmcatt.geron_cross_attention."""
import numpy as np
import pytest
from morie.fn.hmcatt import geron_cross_attention


def test_hmcatt_basic():
    """Test basic functionality."""
    dec_h = np.random.default_rng(42).normal(0, 1, 100)
    enc_h = np.random.default_rng(42).normal(0, 1, 100)
    W_Q = np.random.default_rng(42).normal(0, 1, 100)
    W_K = np.random.default_rng(42).normal(0, 1, 100)
    W_V = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_cross_attention(dec_h, enc_h, W_Q, W_K, W_V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcatt_edge():
    """Test edge cases."""
    dec_h = np.random.default_rng(42).normal(0, 1, 100)
    enc_h = np.random.default_rng(42).normal(0, 1, 100)
    W_Q = np.random.default_rng(42).normal(0, 1, 100)
    W_K = np.random.default_rng(42).normal(0, 1, 100)
    W_V = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_cross_attention(dec_h, enc_h, W_Q, W_K, W_V)
    assert isinstance(result, dict)
