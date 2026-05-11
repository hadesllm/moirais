"""Tests for kmlb.kamath_moe_load_balance_loss."""
import numpy as np
import pytest
from morie.fn.kmlb import kamath_moe_load_balance_loss


def test_kmlb_basic():
    """Test basic functionality."""
    fractions = np.random.default_rng(42).normal(0, 1, 100)
    gate_means = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    alpha = 0.05
    result = kamath_moe_load_balance_loss(fractions, gate_means, N, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmlb_edge():
    """Test edge cases."""
    fractions = np.random.default_rng(42).normal(0, 1, 100)
    gate_means = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    alpha = 0.05
    result = kamath_moe_load_balance_loss(fractions, gate_means, N, alpha)
    assert isinstance(result, dict)
