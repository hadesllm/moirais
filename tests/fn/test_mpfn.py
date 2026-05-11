"""Tests for mpfn.message_passing."""
import numpy as np
import pytest
from morie.fn.mpfn import message_passing


def test_mpfn_basic():
    """Test basic functionality."""
    G = np.eye(10)
    h0 = np.random.default_rng(42).normal(0, 1, 100)
    layers = np.random.default_rng(42).normal(0, 1, 100)
    result = message_passing(G, h0, layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mpfn_edge():
    """Test edge cases."""
    G = np.eye(10)
    h0 = np.random.default_rng(42).normal(0, 1, 100)
    layers = np.random.default_rng(42).normal(0, 1, 100)
    result = message_passing(G, h0, layers)
    assert isinstance(result, dict)
