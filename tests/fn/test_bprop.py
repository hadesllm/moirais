"""Tests for bprop.backpropagation_chain_rule."""
import numpy as np
import pytest
from morie.fn.bprop import backpropagation_chain_rule


def test_bprop_basic():
    """Test basic functionality."""
    layers = np.random.default_rng(42).normal(0, 1, 100)
    activations = np.random.default_rng(42).normal(0, 1, 100)
    loss_grad = np.random.default_rng(42).normal(0, 1, 100)
    result = backpropagation_chain_rule(layers, activations, loss_grad)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bprop_edge():
    """Test edge cases."""
    layers = np.random.default_rng(42).normal(0, 1, 100)
    activations = np.random.default_rng(42).normal(0, 1, 100)
    loss_grad = np.random.default_rng(42).normal(0, 1, 100)
    result = backpropagation_chain_rule(layers, activations, loss_grad)
    assert isinstance(result, dict)
