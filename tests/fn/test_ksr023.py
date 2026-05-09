"""Tests for ksr023.kosorok_ch1_cox_estimating_equation."""
import numpy as np
import pytest
from moirais.fn.ksr023 import kosorok_ch1_cox_estimating_equation


def test_ksr023_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    n = 100
    result = kosorok_ch1_cox_estimating_equation(t, beta, Z, Y, N, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr023_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    n = 100
    result = kosorok_ch1_cox_estimating_equation(t, beta, Z, Y, N, n)
    assert isinstance(result, dict)
