"""Tests for b107.burkov_lm_ch1_layer2_output."""
import numpy as np
import pytest
from moirais.fn.b107 import burkov_lm_ch1_layer2_output


def test_b107_basic():
    """Test basic functionality."""
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    y_1 = np.random.default_rng(42).normal(0, 1, 100)
    b_2_1 = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_layer2_output(W_2, y_1, b_2_1, phi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b107_edge():
    """Test edge cases."""
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    y_1 = np.random.default_rng(42).normal(0, 1, 100)
    b_2_1 = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_layer2_output(W_2, y_1, b_2_1, phi)
    assert isinstance(result, dict)
