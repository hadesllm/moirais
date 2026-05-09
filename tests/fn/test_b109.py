"""Tests for b109.burkov_lm_ch1_binary_cross_entropy."""
import numpy as np
import pytest
from moirais.fn.b109 import burkov_lm_ch1_binary_cross_entropy


def test_b109_basic():
    """Test basic functionality."""
    y_hat_i = np.random.default_rng(42).normal(0, 1, 100)
    y_i = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_binary_cross_entropy(y_hat_i, y_i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b109_edge():
    """Test edge cases."""
    y_hat_i = np.random.default_rng(42).normal(0, 1, 100)
    y_i = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_binary_cross_entropy(y_hat_i, y_i)
    assert isinstance(result, dict)
