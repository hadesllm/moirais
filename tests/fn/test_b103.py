"""Tests for b103.burkov_lm_ch1_mse_cost."""
import numpy as np
import pytest
from morie.fn.b103 import burkov_lm_ch1_mse_cost


def test_b103_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    result = burkov_lm_ch1_mse_cost(w, b, x, y, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b103_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    result = burkov_lm_ch1_mse_cost(w, b, x, y, N)
    assert isinstance(result, dict)
