"""Tests for moeswt.moe_switch_routing."""
import numpy as np
import pytest
from moirais.fn.moeswt import moe_switch_routing


def test_moeswt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    experts = np.random.default_rng(42).normal(0, 1, 100)
    capacity = np.random.default_rng(42).normal(0, 1, 100)
    result = moe_switch_routing(y, x, W_g, experts, capacity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_moeswt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    experts = np.random.default_rng(42).normal(0, 1, 100)
    capacity = np.random.default_rng(42).normal(0, 1, 100)
    result = moe_switch_routing(y, x, W_g, experts, capacity)
    assert isinstance(result, dict)
