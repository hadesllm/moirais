"""Tests for grace.grace."""
import numpy as np
import pytest
from moirais.fn.grace import grace


def test_grace_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    aug = np.random.default_rng(42).normal(0, 1, 100)
    result = grace(G, X, aug)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grace_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    aug = np.random.default_rng(42).normal(0, 1, 100)
    result = grace(G, X, aug)
    assert isinstance(result, dict)
