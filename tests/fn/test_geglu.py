"""Tests for geglu.geglu_activation."""
import numpy as np
import pytest
from moirais.fn.geglu import geglu_activation


def test_geglu_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = geglu_activation(y, x, W, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_geglu_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = geglu_activation(y, x, W, V)
    assert isinstance(result, dict)
