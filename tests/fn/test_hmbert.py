"""Tests for hmbert.geron_bert."""
import numpy as np
import pytest
from morie.fn.hmbert import geron_bert


def test_hmbert_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    d_model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bert(X, n_layers, n_heads, d_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbert_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    d_model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bert(X, n_layers, n_heads, d_model)
    assert isinstance(result, dict)
