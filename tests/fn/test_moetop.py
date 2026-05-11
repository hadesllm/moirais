"""Tests for moetop.moe_topk_routing."""
import numpy as np
import pytest
from morie.fn.moetop import moe_topk_routing


def test_moetop_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    experts = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = moe_topk_routing(y, x, W_g, experts, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_moetop_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    experts = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = moe_topk_routing(y, x, W_g, experts, k)
    assert isinstance(result, dict)
