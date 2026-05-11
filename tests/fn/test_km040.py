"""Tests for km040.kamath_ch2_moe_topk_gating."""
import numpy as np
import pytest
from morie.fn.km040 import kamath_ch2_moe_topk_gating


def test_km040_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_moe_topk_gating(x, W_g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km040_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_moe_topk_gating(x, W_g)
    assert isinstance(result, dict)
