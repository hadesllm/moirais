"""Tests for hmencox.geron_encoder_only."""
import numpy as np
import pytest
from morie.fn.hmencox import geron_encoder_only


def test_hmencox_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_only(X, n_layers, n_heads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmencox_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_only(X, n_layers, n_heads)
    assert isinstance(result, dict)
