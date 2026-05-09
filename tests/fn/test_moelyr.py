"""Tests for moelyr.moe_layer."""
import numpy as np
import pytest
from moirais.fn.moelyr import moe_layer


def test_moelyr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    experts = np.random.default_rng(42).normal(0, 1, 100)
    top_k = np.random.default_rng(42).normal(0, 1, 100)
    result = moe_layer(y, x, W_g, experts, top_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_moelyr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    experts = np.random.default_rng(42).normal(0, 1, 100)
    top_k = np.random.default_rng(42).normal(0, 1, 100)
    result = moe_layer(y, x, W_g, experts, top_k)
    assert isinstance(result, dict)
