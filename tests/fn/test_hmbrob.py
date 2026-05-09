"""Tests for hmbrob.geron_roberta."""
import numpy as np
import pytest
from moirais.fn.hmbrob import geron_roberta


def test_hmbrob_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_roberta(X, n_layers, n_heads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbrob_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_roberta(X, n_layers, n_heads)
    assert isinstance(result, dict)
