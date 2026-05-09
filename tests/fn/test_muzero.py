"""Tests for muzero.muzero."""
import numpy as np
import pytest
from moirais.fn.muzero import muzero


def test_muzero_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    net = np.random.default_rng(42).normal(0, 1, 100)
    unroll_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = muzero(env, net, unroll_steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_muzero_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    net = np.random.default_rng(42).normal(0, 1, 100)
    unroll_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = muzero(env, net, unroll_steps)
    assert isinstance(result, dict)
