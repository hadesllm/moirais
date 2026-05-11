"""Tests for hmmlpf.geron_mlp."""
import numpy as np
import pytest
from morie.fn.hmmlpf import geron_mlp


def test_hmmlpf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    biases = np.random.default_rng(42).normal(0, 1, 100)
    activations = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mlp(X, weights, biases, activations)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmlpf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    biases = np.random.default_rng(42).normal(0, 1, 100)
    activations = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mlp(X, weights, biases, activations)
    assert isinstance(result, dict)
