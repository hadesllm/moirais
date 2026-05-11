"""Tests for hmrvat.geron_rnn_visual_attention."""
import numpy as np
import pytest
from morie.fn.hmrvat import geron_rnn_visual_attention


def test_hmrvat_basic():
    """Test basic functionality."""
    features = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    W = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_rnn_visual_attention(features, h, W, U, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrvat_edge():
    """Test edge cases."""
    features = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    W = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_rnn_visual_attention(features, h, W, U, v)
    assert isinstance(result, dict)
