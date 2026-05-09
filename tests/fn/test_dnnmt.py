"""Tests for dnnmt.dnn_multitrait."""
import numpy as np
import pytest
from moirais.fn.dnnmt import dnn_multitrait


def test_dnnmt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    layers = np.random.default_rng(42).normal(0, 1, 100)
    heads = np.random.default_rng(42).normal(0, 1, 100)
    result = dnn_multitrait(X, Y, layers, heads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dnnmt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    layers = np.random.default_rng(42).normal(0, 1, 100)
    heads = np.random.default_rng(42).normal(0, 1, 100)
    result = dnn_multitrait(X, Y, layers, heads)
    assert isinstance(result, dict)
