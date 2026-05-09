"""Tests for kmmoe.kamath_moe_router_softmax."""
import numpy as np
import pytest
from moirais.fn.kmmoe import kamath_moe_router_softmax


def test_kmmoe_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    Wr = np.random.default_rng(42).normal(0, 1, 100)
    experts = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_moe_router_softmax(x, Wr, experts, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmmoe_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    Wr = np.random.default_rng(42).normal(0, 1, 100)
    experts = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_moe_router_softmax(x, Wr, experts, k)
    assert isinstance(result, dict)
