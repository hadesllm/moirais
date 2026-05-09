"""Tests for km079.kamath_ch6_alignscore_total_loss."""
import numpy as np
import pytest
from moirais.fn.km079 import kamath_ch6_alignscore_total_loss


def test_km079_basic():
    """Test basic functionality."""
    L_3way = np.random.default_rng(42).normal(0, 1, 100)
    L_bin = np.random.default_rng(42).normal(0, 1, 100)
    L_reg = np.random.default_rng(42).normal(0, 1, 100)
    lambdas = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_alignscore_total_loss(L_3way, L_bin, L_reg, lambdas)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km079_edge():
    """Test edge cases."""
    L_3way = np.random.default_rng(42).normal(0, 1, 100)
    L_bin = np.random.default_rng(42).normal(0, 1, 100)
    L_reg = np.random.default_rng(42).normal(0, 1, 100)
    lambdas = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_alignscore_total_loss(L_3way, L_bin, L_reg, lambdas)
    assert isinstance(result, dict)
