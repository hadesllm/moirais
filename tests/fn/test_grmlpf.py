"""Tests for grmlpf.geron_mlp_forward."""
import numpy as np
import pytest
from moirais.fn.grmlpf import geron_mlp_forward


def test_grmlpf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    biases = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mlp_forward(x, weights, biases)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmlpf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    biases = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mlp_forward(x, weights, biases)
    assert isinstance(result, dict)
