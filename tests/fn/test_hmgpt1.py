"""Tests for hmgpt1.geron_gpt1."""
import numpy as np
import pytest
from moirais.fn.hmgpt1 import geron_gpt1


def test_hmgpt1_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gpt1(X, n_layers, n_heads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgpt1_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gpt1(X, n_layers, n_heads)
    assert isinstance(result, dict)
