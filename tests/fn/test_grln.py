"""Tests for grln.geron_layer_normalization."""
import numpy as np
import pytest
from moirais.fn.grln import geron_layer_normalization


def test_grln_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    gamma = 1.0
    beta = 0.8
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_layer_normalization(X, gamma, beta, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grln_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    gamma = 1.0
    beta = 0.8
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_layer_normalization(X, gamma, beta, eps)
    assert isinstance(result, dict)
