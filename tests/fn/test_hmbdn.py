"""Tests for hmbdn.geron_bahdanau_attention."""
import numpy as np
import pytest
from morie.fn.hmbdn import geron_bahdanau_attention


def test_hmbdn_basic():
    """Test basic functionality."""
    h = 0.3
    s_prev = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_bahdanau_attention(h, s_prev, W, U, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbdn_edge():
    """Test edge cases."""
    h = 0.3
    s_prev = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_bahdanau_attention(h, s_prev, W, U, v)
    assert isinstance(result, dict)
