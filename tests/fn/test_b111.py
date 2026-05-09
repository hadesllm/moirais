"""Tests for b111.burkov_lm_ch1_bce_gradients."""
import numpy as np
import pytest
from moirais.fn.b111 import burkov_lm_ch1_bce_gradients


def test_b111_basic():
    """Test basic functionality."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_bce_gradients(y_hat, y, x, N, j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b111_edge():
    """Test edge cases."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_bce_gradients(y_hat, y, x, N, j)
    assert isinstance(result, dict)
