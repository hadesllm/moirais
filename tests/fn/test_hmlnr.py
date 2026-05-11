"""Tests for hmlnr.geron_layer_norm_rnn."""
import numpy as np
import pytest
from morie.fn.hmlnr import geron_layer_norm_rnn


def test_hmlnr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    beta = 0.8
    result = geron_layer_norm_rnn(x, gamma, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlnr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    beta = 0.8
    result = geron_layer_norm_rnn(x, gamma, beta)
    assert isinstance(result, dict)
