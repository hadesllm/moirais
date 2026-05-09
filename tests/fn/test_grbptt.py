"""Tests for grbptt.geron_backprop_through_time."""
import numpy as np
import pytest
from moirais.fn.grbptt import geron_backprop_through_time


def test_grbptt_basic():
    """Test basic functionality."""
    loss_grads = np.random.default_rng(42).normal(0, 1, 100)
    hiddens = np.random.default_rng(42).normal(0, 1, 100)
    inputs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_backprop_through_time(loss_grads, hiddens, inputs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbptt_edge():
    """Test edge cases."""
    loss_grads = np.random.default_rng(42).normal(0, 1, 100)
    hiddens = np.random.default_rng(42).normal(0, 1, 100)
    inputs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_backprop_through_time(loss_grads, hiddens, inputs)
    assert isinstance(result, dict)
