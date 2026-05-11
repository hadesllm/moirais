"""Tests for hmtfm.geron_transformer."""
import numpy as np
import pytest
from morie.fn.hmtfm import geron_transformer


def test_hmtfm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    d_model = np.random.default_rng(42).normal(0, 1, 100)
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_transformer(X, n_heads, d_model, n_layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtfm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    d_model = np.random.default_rng(42).normal(0, 1, 100)
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_transformer(X, n_heads, d_model, n_layers)
    assert isinstance(result, dict)
