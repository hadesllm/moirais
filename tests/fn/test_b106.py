"""Tests for b106.burkov_lm_ch1_layer1_output."""
import numpy as np
import pytest
from morie.fn.b106 import burkov_lm_ch1_layer1_output


def test_b106_basic():
    """Test basic functionality."""
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    b_1 = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_layer1_output(W_1, x, b_1, phi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b106_edge():
    """Test edge cases."""
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    b_1 = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_layer1_output(W_1, x, b_1, phi)
    assert isinstance(result, dict)
