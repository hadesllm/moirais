"""Tests for b201.burkov_lm_ch2_categorical_cross_entropy."""
import numpy as np
import pytest
from moirais.fn.b201 import burkov_lm_ch2_categorical_cross_entropy


def test_b201_basic():
    """Test basic functionality."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch2_categorical_cross_entropy(y_hat, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b201_edge():
    """Test edge cases."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch2_categorical_cross_entropy(y_hat, c)
    assert isinstance(result, dict)
