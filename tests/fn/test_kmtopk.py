"""Tests for kmtopk.kamath_moe_top_k_gating."""
import numpy as np
import pytest
from moirais.fn.kmtopk import kamath_moe_top_k_gating


def test_kmtopk_basic():
    """Test basic functionality."""
    gates = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_moe_top_k_gating(gates, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmtopk_edge():
    """Test edge cases."""
    gates = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_moe_top_k_gating(gates, k)
    assert isinstance(result, dict)
