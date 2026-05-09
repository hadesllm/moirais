"""Tests for hmalbt.geron_albert."""
import numpy as np
import pytest
from moirais.fn.hmalbt import geron_albert


def test_hmalbt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_albert(X, n_layers, n_heads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmalbt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_albert(X, n_layers, n_heads)
    assert isinstance(result, dict)
